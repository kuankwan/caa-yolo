import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from matplotlib.ticker import NullFormatter
num_sum = 250  # 根据样本的实际情况自己改数值
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
    source = plt.scatter(src[:num_sum,0], src[:num_sum,1], s=10, c="#C05757", linewidths=0.1, marker='o', edgecolors='k')  # 绘制散点图。
    target = plt.scatter(tgt[:num_sum,0], tgt[:num_sum,1], s=10, c="#3939EF", linewidths=0.1, marker='o', edgecolors='k')  # 绘制散点图。
    ax.xaxis.set_major_formatter(NullFormatter())  # 设置标签显示格式为空
    ax.yaxis.set_major_formatter(NullFormatter())
    plt.legend((source, target), ('source', 'target'))
    # plt.show()
    save_path = "./tsne_result/" + save_name + ".jpg"
    print("==================finish tsne=====================")
    plt.savefig(save_path)  # 保存图像
    plt.close()


def vis_tsne(class_out, target_out,save_name):
    # class_out是需要t-SNE可视化的特征，可以来自模型任意一层，我这里用的是最后一层
    class_out = class_out[:, :, :, :]
    target_out = target_out[:, :, :, :]
    b,c,w,h = class_out.size()
    b1,c1,w1,h1 = target_out.size()
    src_out = class_out.contiguous().detach().reshape(b,c,w*h)
    tgt_out = target_out.contiguous().detach().reshape(b1,c1,w1*h1)
    source_outs = torch.mean(src_out,dim=2)
    target_outs = torch.mean(tgt_out,dim=2)
    src_feats.append(source_outs)
    tgt_feats.append(target_outs)
    if len(src_feats) == num_sum and len(tgt_feats) == num_sum:
        source_outs = [out.view(-1,c) for out in src_feats]
        target_outs = [out.view(-1, c) for out in tgt_feats]
        source_out = torch.cat(source_outs,dim=0)
        target_out = torch.cat(target_outs, dim=0)
        print(source_out.size())
        src_pred = T_SNE(np.array(source_out.cpu()))
        tgt_pred = T_SNE(np.array(target_out.cpu()))
        draw(src_pred,tgt_pred,save_name)
    if len(src_feats) >= num_sum and len(tgt_feats) >= num_sum:
        src_feats.clear()
        tgt_feats.clear()