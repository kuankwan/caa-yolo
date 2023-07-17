# Copyright (c) OpenMMLab. All rights reserved.
from collections import OrderedDict

import numpy as np
import torch
from matplotlib import pyplot as plt
from mmdet.models.detectors.single_stage import SingleStageDetector
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from mmengine.dist import get_world_size
from mmengine.logging import print_log
from mmengine.optim import OptimWrapper
from sklearn.manifold import TSNE
from torch import Tensor, nn
from mmyolo.registry import MODELS
from typing import List, Tuple, Union,Dict
from mmdet.structures import DetDataSample, OptSampleList, SampleList
from mmyolo.models.losses import sigmoid_focal_loss
from mmyolo.models.da.discriminator import D_layer2,D_layer3,D_layer4,Local_D,Global_D
from mmengine.dist import get_dist_info
import torch.nn.functional as F
from matplotlib.ticker import NullFormatter
from mmdet.models.losses.gfocal_loss import quality_focal_loss_tensor_target
ForwardResults = Union[Dict[str, torch.Tensor], List[DetDataSample],
                       Tuple[torch.Tensor], torch.Tensor]

idx = 0
num_sum = 125  # 根据样本的实际情况自己改数值
src_feats = []
tgt_feats = []
colors = ["#C05757", "#3939EF"]  # ,"g","y","o"#根据情况自己改配色

def T_SNE(pred):
    pred1 = TSNE(n_components=2, init="pca", random_state=0).fit_transform(pred)
    pred1_min,pred1_max = pred1.min(0),pred1.max(0)
    pred1 = (pred1 - pred1_min) / (pred1_max - pred1_min)
    return pred1

def draw(src,tgt,save_name):
    ax = plt.subplot(111)
    my_font1 = {"family": "Times New Roman", "size": 12, "style": "normal","weight":"bold"}
    source = plt.scatter(src[:,0], src[:,1], s=10, c="#C05757", linewidths=0.1, marker='o', edgecolors='k')  # 绘制散点图。
    target = plt.scatter(tgt[:,0], tgt[:,1], s=10, c="#3939EF", linewidths=0.1, marker='o', edgecolors='k')  # 绘制散点图。
    ax.xaxis.set_major_formatter(NullFormatter())  # 设置标签显示格式为空
    ax.yaxis.set_major_formatter(NullFormatter())
    plt.legend((source, target), ('source', 'target'),prop=my_font1)
    # plt.show()
    save_path = "./tsne_result/" + save_name + ".jpg"
    print("==================finish save "+save_path+"=====================")
    plt.savefig(save_path,dpi=300,bbox_inches='tight')  # 保存图像
    plt.close()


def vis_tsne(class_out, target_out,save_name):
    # class_out是需要t-SNE可视化的特征，可以来自模型任意一层，我这里用的是最后一层
    class_out = class_out[:, :, :, :]
    target_out = target_out[:, :, :, :]
    b,c,w,h = class_out.size()
    b1,c1,w1,h1 = target_out.size()
    src_out = class_out.contiguous().detach()
    tgt_out = target_out.contiguous().detach()
    source_outs = torch.mean(src_out,dim=1)
    target_outs = torch.mean(tgt_out,dim=1)
    src_feats.append(source_outs)
    tgt_feats.append(target_outs)
    if len(src_feats) == num_sum and len(tgt_feats) == num_sum:
        source_outs = [out.view(b,-1) for out in src_feats]
        target_outs = [out.view(b1, -1) for out in tgt_feats]
        source_out = torch.cat(source_outs,dim=0)
        target_out = torch.cat(target_outs, dim=0)
        print(source_out.size())
        src_pred = T_SNE(np.array(source_out.cpu()))
        tgt_pred = T_SNE(np.array(target_out.cpu()))
        draw(src_pred,tgt_pred,save_name)
    if len(src_feats) >= num_sum and len(tgt_feats) >= num_sum:
        src_feats.clear()
        tgt_feats.clear()

@MODELS.register_module()
class YOLODetector(SingleStageDetector):
    r"""Implementation of YOLO Series

    Args:
        backbone (:obj:`ConfigDict` or dict): The backbone config.
        neck (:obj:`ConfigDict` or dict): The neck config.
        bbox_head (:obj:`ConfigDict` or dict): The bbox head config.
        train_cfg (:obj:`ConfigDict` or dict, optional): The training config
            of YOLO. Defaults to None.
        test_cfg (:obj:`ConfigDict` or dict, optional): The testing config
            of YOLO. Defaults to None.
        data_preprocessor (:obj:`ConfigDict` or dict, optional): Config of
            :class:`DetDataPreprocessor` to process the input data.
            Defaults to None.
        init_cfg (:obj:`ConfigDict` or list[:obj:`ConfigDict`] or dict or
            list[dict], optional): Initialization config dict.
            Defaults to None.
        use_syncbn (bool): whether to use SyncBatchNorm. Defaults to True.
    """

    def __init__(self,
                 backbone: ConfigType,
                 neck: ConfigType,
                 bbox_head: ConfigType,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None,
                 use_syncbn: bool = True):
        super().__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)

        #Teacher
        self.t_backbone = MODELS.build(backbone)
        if neck is not None:
            self.t_neck = MODELS.build(neck)
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.t_head = MODELS.build(bbox_head)
        # self.teacher_model = nn.Sequential(*self.t_backbone,*self.t_head,*self.t_head)


        self.pga = torch.nn.ModuleList()
        factor:float = backbone['widen_factor']
        self.pga.append(D_layer2(int(neck['in_channels'][0] * factor)))
        self.pga.append(D_layer3(int(neck['in_channels'][1] * factor)))
        self.pga.append(D_layer4(int(neck['in_channels'][2] * factor)))
        self.global_D = nn.ModuleDict()
        self.local_D = nn.ModuleDict()
        self.class_num = bbox_head['head_module']['num_classes']
        for j in range(self.class_num):
            self.local_head = nn.ModuleList()
            self.local_head1 = Local_D(int(bbox_head['head_module']['in_channels'][0] * factor))
            self.local_head.append(self.local_head1)
            self.local_head2 = Local_D(int(bbox_head['head_module']['in_channels'][1] * factor))
            self.local_head.append(self.local_head2)
            self.local_head3 = Local_D(int(bbox_head['head_module']['in_channels'][2] * factor))
            self.local_head.append(self.local_head3)
            name = 'local_head_' + str(j)
            self.local_D.add_module(name=name,module=self.local_head)

            self.global_head = nn.ModuleList()
            self.global_head1 = Global_D(int(bbox_head['head_module']['in_channels'][0] * factor))
            self.global_head.append(self.global_head1)
            self.global_head2 = Global_D(int(bbox_head['head_module']['in_channels'][1] * factor))
            self.global_head.append(self.global_head2)
            self.global_head3 = Global_D(int(bbox_head['head_module']['in_channels'][2] * factor))
            self.global_head.append(self.global_head3)
            name = 'global_head_' + str(j)
            self.global_D.add_module(name=name, module=self.global_head)
        self.theta = 20.0
        self.caa = 8000
        self.copy_teacher = 80000
        self.pga_weigth = 1.0
        self.lcaa_weight = 1.0
        self.gcaa = 1.0
        self.idx = 0


        # TODO： Waiting for mmengine support
        if use_syncbn and get_world_size() > 1:
            torch.nn.SyncBatchNorm.convert_sync_batchnorm(self)
            print_log('Using SyncBatchNorm()', 'current')

    @torch.no_grad()
    def _copy_main_model(self,teacher, student):
        # initialize all parameters
        if get_world_size() > 1:
            rename_model_dict = {
                key[7:]: value for key, value in student.state_dict().items()
            }
            new_model = OrderedDict()
            for key, value in rename_model_dict.items():
                if key in teacher.keys():
                    new_model[key] = rename_model_dict[key]
            teacher.load_state_dict(new_model)
            # model_teacher.load_state_dict(rename_model_dict)
        else:
            new_model = OrderedDict()
            for key, value in student.state_dict().items():
                if key in teacher.state_dict().keys():
                    new_model[key] = value
            teacher.load_state_dict(new_model)
        print("====================================finish copy teacher model=============================")
        return teacher

    @torch.no_grad()
    def _update_teacher_model(self,teacher,student, keep_rate=0.996):
        if get_world_size() > 1:
            student_model_dict = {
                key[7:]: value for key, value in student.state_dict().items()
            }
        else:
            student_model_dict = student.state_dict()

        new_teacher_dict = OrderedDict()
        for key, value in teacher.state_dict().items():
            if key in student_model_dict.keys():
                new_teacher_dict[key] = (
                        student_model_dict[key] *
                        (1 - keep_rate) + value * keep_rate
                )
            else:
                raise Exception("{} is not found in student model".format(key))

        teacher.load_state_dict(new_teacher_dict)
        # print("====================================update teacher model=============================")
        return teacher


    def _train_step(self, data: Union[dict, tuple, list],
                   tar_data: Union[dict, tuple, list],
                   weak_data: Union[dict, tuple, list],
                   optim_wrapper: OptimWrapper,iter:int) -> Dict[str, torch.Tensor]:
        """Implements the default model training process including
        preprocessing, model forward propagation, loss calculation,
        optimization, and back-propagation.

        During non-distributed training. If subclasses do not override the
        :meth:`train_step`, :class:`EpochBasedTrainLoop` or
        :class:`IterBasedTrainLoop` will call this method to update model
        parameters. The default parameter update process is as follows:

        1. Calls ``self.data_processor(data, training=False)`` to collect
           batch_inputs and corresponding data_samples(labels).
        2. Calls ``self(batch_inputs, data_samples, mode='loss')`` to get raw
           loss
        3. Calls ``self.parse_losses`` to get ``parsed_losses`` tensor used to
           backward and dict of loss tensor used to log messages.
        4. Calls ``optim_wrapper.update_params(loss)`` to update model.

        Args:
            data (dict or tuple or list): Data sampled from dataset.
            optim_wrapper (OptimWrapper): OptimWrapper instance
                used to update model parameters.

        Returns:
            Dict[str, torch.Tensor]: A ``dict`` of tensor for logging.
        """
        super().train_step(data=data,optim_wrapper=optim_wrapper)
        if iter == self.copy_teacher:
            self.t_backbone = self._copy_main_model(self.t_backbone,self.backbone)
            self.t_neck = self._copy_main_model(self.t_neck, self.neck)
            self.t_head = self._copy_main_model(self.t_head, self.bbox_head)
        # Enable automatic mixed precision training context.
        with optim_wrapper.optim_context(self):
            data = self.data_preprocessor(data, True)
            losses = self._run_forward(data, mode='loss')  # type: ignore
            tar_data = self.data_preprocessor(tar_data, True)
            weak_data = self.data_preprocessor(weak_data, True)
            if iter > -1:
                da_losses = self.da_loss(data['inputs'], tar_data['inputs'],data['data_samples'],iter)
                losses.update(da_losses)
            if iter > self.copy_teacher:
                unsup_losses = self.unsup_loss(tar_data['inputs'],weak_data['inputs'],data['data_samples'])
                losses.update(unsup_losses)
        parsed_losses, log_vars = self.parse_losses(losses)  # type: ignore
        optim_wrapper.update_params(parsed_losses)
        if iter > self.copy_teacher:
            self.t_backbone = self._update_teacher_model(self.t_backbone, self.backbone)
            self.t_neck = self._update_teacher_model(self.t_neck, self.neck)
            self.t_head = self._update_teacher_model(self.t_head, self.bbox_head)
        return log_vars

    def run_forward(self, data: Union[dict, tuple, list],tar_data: Union[dict, tuple, list],
                     mode: str) -> Union[Dict[str, torch.Tensor], list]:
        """Unpacks data for :meth:`forward`

        Args:
            data (dict or tuple or list): Data sampled from dataset.
            mode (str): Mode of forward.

        Returns:
            dict or list: Results of training or testing mode.
        """
        if isinstance(data, dict) :
            results = self(**data, mode=mode)
        elif isinstance(data, (list, tuple)):
            results = self(*data, mode=mode)
        else:
            raise TypeError('Output of `data_preprocessor` should be '
                            f'list, tuple or dict, but got {type(data)}')
        return results

    def forward(self,
                inputs: torch.Tensor,
                data_samples: OptSampleList = None,
                mode: str = 'tensor') -> ForwardResults:
        """The unified entry for a forward process in both training and test.

        The method should accept three modes: "tensor", "predict" and "loss":

        - "tensor": Forward the whole network and return tensor or tuple of
        tensor without any post-processing, same as a common nn.Module.
        - "predict": Forward and return the predictions, which are fully
        processed to a list of :obj:`DetDataSample`.
        - "loss": Forward and return a dict of losses according to the given
        inputs and data samples.

        Note that this method doesn't handle either back propagation or
        parameter update, which are supposed to be done in :meth:`train_step`.

        Args:
            inputs (torch.Tensor): The input tensor with shape
                (N, C, ...) in general.
            data_samples (list[:obj:`DetDataSample`], optional): A batch of
                data samples that contain annotations and predictions.
                Defaults to None.
            mode (str): Return what kind of value. Defaults to 'tensor'.

        Returns:
            The return type depends on ``mode``.

            - If ``mode="tensor"``, return a tensor or a tuple of tensor.
            - If ``mode="predict"``, return a list of :obj:`DetDataSample`.
            - If ``mode="loss"``, return a dict of tensor.
        """
        # print(inputs.shape)
        if mode == 'loss':
            return self.loss(inputs, data_samples)
        elif mode == 'predict':
            return self.predict(inputs, data_samples)
        elif mode == 'tensor':
            return self._forward(inputs, data_samples)
        else:
            raise RuntimeError(f'Invalid mode "{mode}". '
                               'Only supports loss, predict and tensor mode')


    def _forward(
            self,
            batch_inputs: Tensor,
            batch_data_samples: OptSampleList = None) -> Tuple[List[Tensor]]:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

         Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (list[:obj:`DetDataSample`]): Each item contains
                the meta information of each image and corresponding
                annotations.

        Returns:
            tuple[list]: A tuple of features from ``bbox_head`` forward.
        """
        backbone_feats, neck_feats = self.extract_feat(batch_inputs)
        results = self.bbox_head.forward(neck_feats)
        return results

    def unsup_loss(self,target_input: Tensor,weak_input: Tensor,batch_data_samples:Union[list,dict]):
        device = target_input.device
        loss_unsup = torch.zeros(1).to(device)
        num = len(batch_data_samples['img_metas'])
        _, world_size = get_dist_info()
        target_outputs = self._forward(target_input)
        with torch.no_grad():
            if self.t_neck is not None:
                weak_outputs = self.t_head(self.t_neck(self.t_backbone(weak_input)))
            else:
                weak_outputs = self.t_head(self.t_backbone(weak_input))
        for idx,(target_out_cls,weak_out_cls,target_out_reg,weak_out_reg,target_out_dist,weak_out_dist) in enumerate(zip(target_outputs[0],weak_outputs[0],target_outputs[1],weak_outputs[1],target_outputs[2],weak_outputs[2])):
            b,c,w,h = target_out_cls.size()
            loss_unsup += quality_focal_loss_tensor_target(target_out_cls.permute(0,2,3,1).contiguous().view(-1,c),
                                                           weak_out_cls.permute(0,2,3,1).contiguous().view(-1,c).sigmoid().detach())
        return dict(loss_unsup=(loss_unsup * num * world_size)/3.)




    def da_loss(self, source_inputs: Tensor,target_input: Tensor,batch_data_samples:Union[list,dict],iter:int) -> Union[dict, list]:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components.
        """

        num = len(batch_data_samples['img_metas'])
        sou_backbone_feats, sou_neck_feats = self.extract_feat(source_inputs)
        tar_backbone_feats, tar_neck_feats = self.extract_feat(target_input)

        assert len(sou_neck_feats) == len(self.pga) and len(tar_neck_feats) == len(self.pga)
        device = sou_backbone_feats[0].device
        pga_loss = torch.zeros(1).to(device)
        bs = sou_backbone_feats[0].size(0)
        need_prop_src = torch.ones(1).to(device)
        need_prop_tgt = torch.zeros(1).to(device)
        for idx in range(len(sou_backbone_feats)):
            src_logit,_ = self.pga[idx](sou_backbone_feats[idx],need_prop_src)
            tgt_logit,_ = self.pga[idx](tar_backbone_feats[idx],need_prop_tgt)
            src_label = torch.full(src_logit.shape, 0.0, dtype=torch.float,device=device)
            tgt_label = torch.full(tgt_logit.shape, 1.0, dtype=torch.float, device=device)
            if idx < len(sou_backbone_feats):
                srcLoss = image_d_loss(src_logit, src_label, device=device,bce=True)
                tgtLoss = image_d_loss(tgt_logit, tgt_label,device=device,bce=True)
            else:
                srcLoss = image_d_loss(src_logit, src_label, device=device, bce=True)
                tgtLoss = image_d_loss(tgt_logit, tgt_label, device=device, bce=True)
            loss = (srcLoss + tgtLoss)
            pga_loss += loss

        _, world_size = get_dist_info()
        losses = dict(loss_da=(pga_loss * num * world_size * self.pga_weigth / 3))
        if sou_neck_feats is not None:
            src_outputs = self.bbox_head.forward(sou_neck_feats)
            # for instance_d, feat in zip(self.instance_da, neck_feats):
            #     instance_logits.append(instance_d(feat))
        else:
            src_outputs = self.bbox_head.forward(sou_backbone_feats)

        if tar_neck_feats is not None:
            tgtoutputs = self.bbox_head.forward(tar_neck_feats)
            # for instance_d, feat in zip(self.instance_da, neck_feats):
            #     instance_logits.append(instance_d(feat))
        else:
            tgtoutputs = self.bbox_head.forward(tar_backbone_feats)

        all_src_local = []
        all_tgt_local = []
        all_src_global = []
        all_tgt_global = []
        if iter > self.caa:
            for idx,(src_cls_pred,src_neck,tgt_cls_pred,tgt_neck) in enumerate(zip(src_outputs[0],sou_neck_feats,tgtoutputs[0],tar_neck_feats)):
                src_local_list = []
                tgt_local_list = []
                src_global_list = []
                tgt_global_list = []
                src_weigth = F.sigmoid(src_cls_pred.sigmoid() * self.theta)
                tgt_weigth = F.sigmoid(tgt_cls_pred.sigmoid() * self.theta)
                for i in range(self.class_num):
                    src_local_input = src_weigth[:,i:i+1,:,:] * src_neck + src_neck
                    tgt_local_input = tgt_weigth[:, i:i + 1, :, :] * tgt_neck + tgt_neck
                    localname = 'local_head_' + str(i)
                    src_local_logit = self.local_D[localname][idx](src_local_input)
                    src_local_list.append(src_local_logit)
                    tgt_local_logit = self.local_D[localname][idx](tgt_local_input)
                    tgt_local_list.append(tgt_local_logit)
                    src_global_weight = 1 + F.sigmoid(src_local_logit) * torch.log(F.sigmoid(src_local_logit))
                    src_global_input = src_global_weight * src_neck + src_neck
                    tgt_global_weight = 1 + F.sigmoid(tgt_local_logit) * torch.log(F.sigmoid(tgt_local_logit))
                    tgt_global_input = tgt_global_weight * tgt_neck + tgt_neck
                    globalname = 'global_head_' + str(i)
                    src_global_logit = self.global_D[globalname][idx](src_global_input)
                    src_global_list.append(src_global_logit)
                    tgt_global_logit = self.global_D[globalname][idx](tgt_global_input)
                    tgt_global_list.append(tgt_global_logit)
                s_local_pred = torch.cat(src_local_list, dim=1)
                t_local_pred = torch.cat(tgt_local_list, dim=1)
                s_global_pred = torch.cat(src_global_list, dim=1)
                t_global_pred = torch.cat(tgt_global_list, dim=1)
                all_src_local.append(s_local_pred)
                all_tgt_local.append(t_local_pred)
                all_src_global.append(s_global_pred)
                all_tgt_global.append(t_global_pred)

            local_loss = local_d_loss(all_src_local,all_tgt_local,device,world_size,num)
            global_loss = global_d_loss(all_src_global, all_tgt_global, device, world_size, num)
            losses.update(local_loss)
            losses.update(global_loss)
            self.idx += 1
            # vis_tsne(sou_backbone_feats[-1], tar_backbone_feats[-1],str(self.idx))

        return losses

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components.
        """
        backbone_feats, neck_feats = self.extract_feat(batch_inputs)
        if neck_feats is not None:
            losses = self.bbox_head.loss(neck_feats, batch_data_samples)
            # for instance_d, feat in zip(self.instance_da, neck_feats):
            #     instance_logits.append(instance_d(feat))
        else:
            losses = self.bbox_head.loss(backbone_feats, batch_data_samples)
        return losses

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool): Whether to rescale the results.
                Defaults to True.

        Returns:
            list[:obj:`DetDataSample`]: Detection results of the
            input images. Each DetDataSample usually contain
            'pred_instances'. And the ``pred_instances`` usually
            contains following keys.

                - scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                    the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        x_b,x = self.extract_feat(batch_inputs)
        if x is  None:
            x = x_b
        results_list = self.bbox_head.predict(
            x, batch_data_samples, rescale=rescale)
        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples

    def extract_feat(self, batch_inputs: Tensor)-> Tuple[any,any]:
        """Extract features.

        Args:
            batch_inputs (Tensor): Image tensor with shape (N, C, H ,W).

        Returns:
            backbone_feat(tuple[Tensor]): Multi-level backbone features that may have
            neck_feat(tuple[Tensor]): Multi-level features that may have
            different resolutions.
        """
        backbone_feats = self.backbone(batch_inputs)
        if self.with_neck:
            neck_feats = self.neck(backbone_feats)
            return backbone_feats, neck_feats
        return backbone_feats,None

def image_d_loss(image_logit: Tensor,label:Tensor,device,bce:bool):
    # image_logit = image_logit.permute(0, 2, 3, 1).view(-1, 1)
    # label = torch.full(image_logit.shape, z, dtype=torch.float,device=device)
    if bce:
        d_loss = sigmoid_focal_loss(image_logit,label,alpha=-1,reduction='mean')
    else:
        d_loss = torch.mean(torch.pow((label-image_logit.sigmoid()),2))
    return d_loss


def local_d_loss(src_pred:list,tgt_pred:list,device,world_size,num):
    local_loss = torch.zeros(1).to(device)
    number = len(src_pred)
    for src_p,tgt_p in zip(src_pred,tgt_pred):
        src_label = torch.full(src_p.shape, 0.0, dtype=torch.float, device=device)
        tgt_label = torch.full(tgt_p.shape, 1.0, dtype=torch.float, device=device)
        l_loss = sigmoid_focal_loss(src_p,src_label,alpha=-1,reduction='mean') + sigmoid_focal_loss(tgt_p,tgt_label,alpha=-1,reduction='mean')
        local_loss += l_loss
    return dict(loss_local=(local_loss * num * world_size / number))

def global_d_loss(src_pred:list,tgt_pred:list,device,world_size,num):
    global_loss = torch.zeros(1).to(device)
    number = len(src_pred)
    # s_global_all = torch.cat(src_pred, dim=1)
    # t_global_all = torch.cat(tgt_pred, dim=1)
    # src_label = torch.full(s_global_all.shape, 0.0, dtype=torch.float, device=device)
    # tgt_label = torch.full(t_global_all.shape, 1.0, dtype=torch.float, device=device)
    # g_loss = F.binary_cross_entropy_with_logits(s_global_all, src_label,
    #                                             reduction='mean') + F.binary_cross_entropy_with_logits(t_global_all, tgt_label,
    #                                                                                                    reduction='mean')
    for src_p,tgt_p in zip(src_pred,tgt_pred):
        src_label = torch.full(src_p.shape, 0.0, dtype=torch.float, device=device)
        tgt_label = torch.full(tgt_p.shape, 1.0, dtype=torch.float, device=device)
        g_loss = F.binary_cross_entropy_with_logits(src_p,src_label,reduction='mean') + F.binary_cross_entropy_with_logits(tgt_p,tgt_label,reduction='mean')
        global_loss += g_loss * 0.5
    return dict(loss_global=(global_loss * num * world_size / number))


