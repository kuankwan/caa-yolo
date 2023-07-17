from mmengine.runner.runner import *
import copy
import logging
import os
import os.path as osp
import pickle
import platform
import time
import warnings
from collections import OrderedDict
from functools import partial
from typing import Callable, Dict, List, Optional, Sequence, Union

import torch
import torch.nn as nn
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from .checkpoint import (_load_checkpoint, _load_checkpoint_to_model,
                         find_latest_checkpoint, get_state_dict,
                         save_checkpoint, weights_to_cpu)
import mmengine
from mmengine.config import Config, ConfigDict
from mmengine.dataset import worker_init_fn
from mmengine.device import get_device
from mmengine.dist import (broadcast, get_dist_info, get_rank, init_dist,
                           is_distributed, master_only)
from mmengine.evaluator import Evaluator
from mmengine.fileio import FileClient, join_path
from mmengine.hooks import Hook
from mmengine.logging import MessageHub, MMLogger, print_log
from mmengine.model import (MMDistributedDataParallel, convert_sync_batchnorm,
                            is_model_wrapper, revert_sync_batchnorm)
from mmengine.optim import (OptimWrapper, OptimWrapperDict, _ParamScheduler,
                            build_optim_wrapper)
from mmengine.registry import (DATA_SAMPLERS, DATASETS, EVALUATOR, FUNCTIONS,
                               HOOKS, LOG_PROCESSORS, LOOPS, MODEL_WRAPPERS,
                               MODELS, OPTIM_WRAPPERS, PARAM_SCHEDULERS,
                               RUNNERS, VISUALIZERS, DefaultScope)
from mmengine.utils import digit_version, get_git_hash, is_seq_of
from mmengine.utils.dl_utils import (TORCH_VERSION, collect_env,
                                     set_multi_processing)
from mmengine.visualization import Visualizer
from mmengine.runner.base_loop import BaseLoop
from mmengine.runner.log_processor import LogProcessor
from mmyolo.engine.runner.loops import EpochBased_TrainLoop, IterBased_TrainLoop, Test_Loop, Val_Loop
from mmengine.runner.priority import Priority, get_priority
from mmengine.runner.utils import set_random_seed
ConfigType = Union[Dict, Config, ConfigDict]
ParamSchedulerType = Union[List[_ParamScheduler], Dict[str,
                                                       List[_ParamScheduler]]]
OptimWrapperType = Union[OptimWrapper, OptimWrapperDict]



class Runner(Runner):
    cfg: Config
    _train_loop: Optional[Union[BaseLoop, Dict]]
    _val_loop: Optional[Union[BaseLoop, Dict]]
    _test_loop: Optional[Union[BaseLoop, Dict]]
    def __init__(
            self,
            model: Union[nn.Module, Dict],
            work_dir: str,
            train_dataloader: Optional[Union[DataLoader, Dict]] = None,
            target_dataloader: Optional[Union[DataLoader, Dict]] = None,
            weak_dataloader: Optional[Union[DataLoader, Dict]] = None,
            val_dataloader: Optional[Union[DataLoader, Dict]] = None,
            test_dataloader: Optional[Union[DataLoader, Dict]] = None,
            train_cfg: Optional[Dict] = None,
            val_cfg: Optional[Dict] = None,
            test_cfg: Optional[Dict] = None,
            auto_scale_lr: Optional[Dict] = None,
            optim_wrapper: Optional[Union[OptimWrapper, Dict]] = None,
            param_scheduler: Optional[Union[_ParamScheduler, Dict, List]] = None,
            val_evaluator: Optional[Union[Evaluator, Dict, List]] = None,
            test_evaluator: Optional[Union[Evaluator, Dict, List]] = None,
            default_hooks: Optional[Dict[str, Union[Hook, Dict]]] = None,
            custom_hooks: Optional[List[Union[Hook, Dict]]] = None,
            data_preprocessor: Union[nn.Module, Dict, None] = None,
            load_from: Optional[str] = None,
            resume: bool = False,
            launcher: str = 'none',
            env_cfg: Dict = dict(dist_cfg=dict(backend='nccl')),
            log_processor: Optional[Dict] = None,
            log_level: str = 'INFO',
            visualizer: Optional[Union[Visualizer, Dict]] = None,
            default_scope: str = 'mmengine',
            randomness: Dict = dict(seed=None),
            experiment_name: Optional[str] = None,
            cfg: Optional[ConfigType] = None,):
        super().__init__(
            model=model,
            work_dir=work_dir,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            test_dataloader=test_dataloader,
            train_cfg=train_cfg,
            val_cfg=val_cfg,
            test_cfg=test_cfg,
            auto_scale_lr=auto_scale_lr,
            optim_wrapper=optim_wrapper,
            param_scheduler=param_scheduler,
            val_evaluator=val_evaluator,
            test_evaluator=test_evaluator,
            default_hooks=default_hooks,
            custom_hooks=custom_hooks,
            data_preprocessor=data_preprocessor,
            load_from=load_from,
            resume=resume,
            launcher=launcher,
            env_cfg=env_cfg,
            log_processor=log_processor,
            log_level=log_level,
            visualizer=visualizer,
            default_scope=default_scope,
            randomness=randomness,
            experiment_name=experiment_name,
            cfg=cfg)

        self._work_dir = osp.abspath(work_dir)
        mmengine.mkdir_or_exist(self._work_dir)

        # recursively copy the `cfg` because `self.cfg` will be modified
        # everywhere.
        if cfg is not None:
            if isinstance(cfg, Config):
                self.cfg = copy.deepcopy(cfg)
            elif isinstance(cfg, dict):
                self.cfg = Config(cfg)
        else:
            self.cfg = Config(dict())

        # lazy initialization
        training_related = [train_dataloader, train_cfg, optim_wrapper]
        if not (all(item is None for item in training_related)
                or all(item is not None for item in training_related)):
            raise ValueError(
                'train_dataloader, train_cfg, and optim_wrapper should be '
                'either all None or not None, but got '
                f'train_dataloader={train_dataloader}, '
                f'train_cfg={train_cfg}, '
                f'optim_wrapper={optim_wrapper}.')
        self._train_dataloader = train_dataloader
        self._train_loop = train_cfg

        training_target_related = [target_dataloader, train_cfg, optim_wrapper]
        if not (all(item is None for item in training_target_related)
                or all(item is not None for item in training_target_related)):
            raise ValueError(
                'target_dataloader, train_cfg, and optim_wrapper should be '
                'either all None or not None, but got '
                f'train_dataloader={target_dataloader}, '
                f'train_cfg={train_cfg}, '
                f'optim_wrapper={optim_wrapper}.')
        self._target_dataloader = target_dataloader
        self._weak_dataloader = weak_dataloader
        self._train_target_loop = train_cfg

        self.optim_wrapper: Optional[Union[OptimWrapper, dict]]
        self.optim_wrapper = optim_wrapper

        self.auto_scale_lr = auto_scale_lr

        # If there is no need to adjust learning rate, momentum or other
        # parameters of optimizer, param_scheduler can be None
        if param_scheduler is not None and self.optim_wrapper is None:
            raise ValueError(
                'param_scheduler should be None when optim_wrapper is None, '
                f'but got {param_scheduler}')

        # Parse `param_scheduler` to a list or a dict. If `optim_wrapper` is a
        # `dict` with single optimizer, parsed param_scheduler will be a
        # list of parameter schedulers. If `optim_wrapper` is
        # a `dict` with multiple optimizers, parsed `param_scheduler` will be
        # dict with multiple list of parameter schedulers.
        self._check_scheduler_cfg(param_scheduler)
        self.param_schedulers = param_scheduler

        val_related = [val_dataloader, val_cfg, val_evaluator]
        if not (all(item is None
                    for item in val_related) or all(item is not None
                                                    for item in val_related)):
            raise ValueError(
                'val_dataloader, val_cfg, and val_evaluator should be either '
                'all None or not None, but got '
                f'val_dataloader={val_dataloader}, val_cfg={val_cfg}, '
                f'val_evaluator={val_evaluator}')
        self._val_dataloader = val_dataloader
        self._val_loop = val_cfg
        self._val_evaluator = val_evaluator

        test_related = [test_dataloader, test_cfg, test_evaluator]
        if not (all(item is None for item in test_related)
                or all(item is not None for item in test_related)):
            raise ValueError(
                'test_dataloader, test_cfg, and test_evaluator should be '
                'either all None or not None, but got '
                f'test_dataloader={test_dataloader}, test_cfg={test_cfg}, '
                f'test_evaluator={test_evaluator}')
        self._test_dataloader = test_dataloader
        self._test_loop = test_cfg
        self._test_evaluator = test_evaluator

        self._launcher = launcher
        if self._launcher == 'none':
            self._distributed = False
        else:
            self._distributed = True

        # self._timestamp will be set in the `setup_env` method. Besides,
        # it also will initialize multi-process and (or) distributed
        # environment.
        self.setup_env(env_cfg)
        # self._deterministic and self._seed will be set in the
        # `set_randomness`` method
        self._randomness_cfg = randomness
        self.set_randomness(**randomness)

        if experiment_name is not None:
            self._experiment_name = f'{experiment_name}_{self._timestamp}'
        elif self.cfg.filename is not None:
            filename_no_ext = osp.splitext(osp.basename(self.cfg.filename))[0]
            self._experiment_name = f'{filename_no_ext}_{self._timestamp}'
        else:
            self._experiment_name = self.timestamp
        self._log_dir = osp.join(self.work_dir, self.timestamp)
        mmengine.mkdir_or_exist(self._log_dir)
        # Used to reset registries location. See :meth:`Registry.build` for
        # more details.
        self.default_scope = DefaultScope.get_instance(
            self._experiment_name, scope_name=default_scope)
        # Build log processor to format message.
        log_processor = dict() if log_processor is None else log_processor
        self.log_processor = self.build_log_processor(log_processor)
        # Since `get_instance` could return any subclass of ManagerMixin. The
        # corresponding attribute needs a type hint.
        self.logger = self.build_logger(log_level=log_level)

        # Collect and log environment information.
        self._log_env(env_cfg)

        # Build `message_hub` for communication among components.
        # `message_hub` can store log scalars (loss, learning rate) and
        # runtime information (iter and epoch). Those components that do not
        # have access to the runner can get iteration or epoch information
        # from `message_hub`. For example, models can get the latest created
        # `message_hub` by
        # `self.message_hub=MessageHub.get_current_instance()` and then get
        # current epoch by `cur_epoch = self.message_hub.get_info('epoch')`.
        # See `MessageHub` and `ManagerMixin` for more details.
        self.message_hub = self.build_message_hub()
        # visualizer used for writing log or visualizing all kinds of data
        self.visualizer = self.build_visualizer(visualizer)
        if self.cfg:
            self.visualizer.add_config(self.cfg)

        self._load_from = load_from
        self._resume = resume
        # flag to mark whether checkpoint has been loaded or resumed
        self._has_loaded = False

        # build a model
        if isinstance(model, dict) and data_preprocessor is not None:
            # Merge the data_preprocessor to model config.
            model.setdefault('data_preprocessor', data_preprocessor)
        self.model = self.build_model(model)
        # wrap model
        self.model = self.wrap_model(
            self.cfg.get('model_wrapper_cfg'), self.model)

        # get model name from the model class
        if hasattr(self.model, 'module'):
            self._model_name = self.model.module.__class__.__name__
        else:
            self._model_name = self.model.__class__.__name__

        self._hooks: List[Hook] = []
        # register hooks to `self._hooks`
        self.register_hooks(default_hooks, custom_hooks)
        # log hooks information
        self.logger.info(f'Hooks will be executed in the following '
                         f'order:\n{self.get_hooks_info()}')

        # dump `cfg` to `work_dir`
        self.dump_config()

    @classmethod
    def from_cfg(cls, cfg: ConfigType) -> 'Runner':
        """Build a runner from config.

        Args:
            cfg (ConfigType): A config used for building runner. Keys of
                ``cfg`` can see :meth:`__init__`.

        Returns:
            Runner: A runner build from ``cfg``.
        """
        cfg = copy.deepcopy(cfg)
        runner = cls(
            model=cfg['model'],
            work_dir=cfg['work_dir'],
            train_dataloader=cfg.get('train_dataloader'),
            target_dataloader=cfg.get('target_dataloader'),
            weak_dataloader=cfg.get('weak_dataloader'),
            val_dataloader=cfg.get('val_dataloader'),
            test_dataloader=cfg.get('test_dataloader'),
            train_cfg=cfg.get('train_cfg'),
            val_cfg=cfg.get('val_cfg'),
            test_cfg=cfg.get('test_cfg'),
            auto_scale_lr=cfg.get('auto_scale_lr'),
            optim_wrapper=cfg.get('optim_wrapper'),
            param_scheduler=cfg.get('param_scheduler'),
            val_evaluator=cfg.get('val_evaluator'),
            test_evaluator=cfg.get('test_evaluator'),
            default_hooks=cfg.get('default_hooks'),
            custom_hooks=cfg.get('custom_hooks'),
            data_preprocessor=cfg.get('data_preprocessor'),
            load_from=cfg.get('load_from'),
            resume=cfg.get('resume', False),
            launcher=cfg.get('launcher', 'none'),
            env_cfg=cfg.get('env_cfg'),  # type: ignore
            log_processor=cfg.get('log_processor'),
            log_level=cfg.get('log_level', 'INFO'),
            visualizer=cfg.get('visualizer'),
            default_scope=cfg.get('default_scope', 'mmengine'),
            randomness=cfg.get('randomness', dict(seed=None)),
            experiment_name=cfg.get('experiment_name'),
            cfg=cfg,
        )

        return runner

    @property
    def train_loop(self):
        """:obj:`BaseLoop`: A loop to run training."""
        if isinstance(self._train_loop, BaseLoop) or self._train_loop is None:
            return self._train_loop
        else:
            self._train_loop = self.build_train_loop(self._train_loop)
            return self._train_loop

    @property
    def target_dataloader(self):
        """The data loader for training."""
        return self.target_dataloader

    def build_train_loop(self, loop: Union[BaseLoop, Dict]) -> BaseLoop:
        """Build training loop.

        Examples of ``loop``::

            # `EpochBasedTrainLoop` will be used
            loop = dict(by_epoch=True, max_epochs=3)

            # `IterBasedTrainLoop` will be used
            loop = dict(by_epoch=False, max_epochs=3)

            # custom training loop
            loop = dict(type='CustomTrainLoop', max_epochs=3)

        Args:
            loop (BaseLoop or dict): A training loop or a dict to build
                training loop. If ``loop`` is a training loop object, just
                returns itself.

        Returns:
            :obj:`BaseLoop`: Training loop object build from ``loop``.
        """
        if isinstance(loop, BaseLoop):
            return loop
        elif not isinstance(loop, dict):
            raise TypeError(
                f'loop should be a Loop object or dict, but got {loop}')

        loop_cfg = copy.deepcopy(loop)

        if 'type' in loop_cfg and 'by_epoch' in loop_cfg:
            raise RuntimeError(
                'Only one of `type` or `by_epoch` can exist in `loop_cfg`.')

        if 'type' in loop_cfg:
            print("----------------------------------使用系统定义loop------------------------------------")
            loop = LOOPS.build(
                loop_cfg,
                default_args=dict(
                    runner=self, dataloader=self._train_dataloader))
        else:
            print("----------------------------------使用自定义loop------------------------------------")
            by_epoch = loop_cfg.pop('by_epoch')
            if by_epoch:
                loop = EpochBased_TrainLoop(
                    **loop_cfg, runner=self, dataloader=self._train_dataloader,tar_dataloader=self._target_dataloader
                    ,weak_dataloader=self._weak_dataloader)
            else:
                loop = IterBased_TrainLoop(
                    **loop_cfg, runner=self, dataloader=self._train_dataloader,tar_dataloader=self._target_dataloader
                    ,weak_dataloader=self._weak_dataloader)
        return loop  # type: ignore

    def build_val_loop(self, loop: Union[BaseLoop, Dict]) -> BaseLoop:
        """Build validation loop.

        Examples of ``loop``:

            # `ValLoop` will be used
            loop = dict()

            # custom validation loop
            loop = dict(type='CustomValLoop')

        Args:
            loop (BaseLoop or dict): A validation loop or a dict to build
                validation loop. If ``loop`` is a validation loop object, just
                returns itself.

        Returns:
            :obj:`BaseLoop`: Validation loop object build from ``loop``.
        """
        if isinstance(loop, BaseLoop):
            return loop
        elif not isinstance(loop, dict):
            raise TypeError(
                f'train_loop should be a Loop object or dict, but got {loop}')

        loop_cfg = copy.deepcopy(loop)

        if 'type' in loop_cfg:
            loop = LOOPS.build(
                loop_cfg,
                default_args=dict(
                    runner=self,
                    dataloader=self._val_dataloader,
                    evaluator=self._val_evaluator))
        else:
            loop = Val_Loop(
                **loop_cfg,
                runner=self,
                dataloader=self._val_dataloader,
                evaluator=self._val_evaluator)  # type: ignore

        return loop  # type: ignore

    def build_test_loop(self, loop: Union[BaseLoop, Dict]) -> BaseLoop:
        """Build test loop.

        Examples of ``loop``::

            # `TestLoop` will be used
            loop = dict()

            # custom test loop
            loop = dict(type='CustomTestLoop')

        Args:
            loop (BaseLoop or dict): A test loop or a dict to build test loop.
                If ``loop`` is a test loop object, just returns itself.

        Returns:
            :obj:`BaseLoop`: Test loop object build from ``loop_cfg``.
        """
        if isinstance(loop, BaseLoop):
            return loop
        elif not isinstance(loop, dict):
            raise TypeError(
                f'train_loop should be a Loop object or dict, but got {loop}')

        loop_cfg = copy.deepcopy(loop)  # type: ignore

        if 'type' in loop_cfg:
            loop = LOOPS.build(
                loop_cfg,
                default_args=dict(
                    runner=self,
                    dataloader=self._test_dataloader,
                    evaluator=self._test_evaluator))
        else:
            loop = Test_Loop(
                **loop_cfg,
                runner=self,
                dataloader=self._test_dataloader,
                evaluator=self._test_evaluator)  # type: ignore

        return loop  # type: ignore

    def call_hook(self, fn_name: str, **kwargs) -> None:
        """Call all hooks.

        Args:
            fn_name (str): The function name in each hook to be called, such as
                "before_train_epoch".
            **kwargs: Keyword arguments passed to hook.
        """
        for hook in self._hooks:
            # support adding additional custom hook methods
            if hasattr(hook, fn_name):
                try:
                    getattr(hook, fn_name)(self, **kwargs)
                except TypeError as e:
                    raise TypeError(f'{e} in {hook}') from None

    def load_checkpoint(self,
                        filename: str,
                        map_location: Union[str, Callable] = 'cpu',
                        strict: bool = False,
                        revise_keys: list = [(r'^module.', '')]):
        """Load checkpoint from given ``filename``.

        Args:
            filename (str): Accept local filepath, URL, ``torchvision://xxx``,
                ``open-mmlab://xxx``.
            map_location (str or callable): A string or a callable function to
                specifying how to remap storage locations.
                Defaults to 'cpu'.
            strict (bool): strict (bool): Whether to allow different params for
                the model and checkpoint.
            revise_keys (list): A list of customized keywords to modify the
                state_dict in checkpoint. Each item is a (pattern, replacement)
                pair of the regular expression operations. Defaults to strip
                the prefix 'module.' by [(r'^module\\.', '')].
        """
        checkpoint = _load_checkpoint(filename, map_location=map_location)

        # Add comments to describe the usage of `after_load_ckpt`
        self.call_hook('after_load_checkpoint', checkpoint=checkpoint)

        if is_model_wrapper(self.model):
            model = self.model.module
        else:
            model = self.model

        checkpoint = _load_checkpoint_to_model(
            model, checkpoint, strict, revise_keys=revise_keys)

        self._has_loaded = True

        self.logger.info(f'Load checkpoint from {filename}')

        return checkpoint

    def train(self) -> nn.Module:
        """Launch training.

        Returns:
            nn.Module: The model after training.
        """
        if is_model_wrapper(self.model):
            ori_model = self.model.module
        else:
            ori_model = self.model
        assert hasattr(ori_model, 'train_step'), (
            'If you want to train your model, please make sure your model '
            'has implemented `train_step`.')

        if self._val_loop is not None:
            assert hasattr(ori_model, 'val_step'), (
                'If you want to validate your model, please make sure your '
                'model has implemented `val_step`.')

        if self._train_loop is None:
            raise RuntimeError(
                '`self._train_loop` should not be None when calling train '
                'method. Please provide `train_dataloader`, `train_cfg`, '
                '`optimizer` and `param_scheduler` arguments when '
                'initializing runner.')

        self._train_loop = self.build_train_loop(
            self._train_loop)  # type: ignore

        # `build_optimizer` should be called before `build_param_scheduler`
        #  because the latter depends on the former
        self.optim_wrapper = self.build_optim_wrapper(self.optim_wrapper)
        # Automatically scaling lr by linear scaling rule
        self.scale_lr(self.optim_wrapper, self.auto_scale_lr)

        if self.param_schedulers is not None:
            self.param_schedulers = self.build_param_scheduler(  # type: ignore
                self.param_schedulers)  # type: ignore

        if self._val_loop is not None:
            self._val_loop = self.build_val_loop(
                self._val_loop)  # type: ignore
        # TODO: add a contextmanager to avoid calling `before_run` many times
        self.call_hook('before_run')

        # initialize the model weights
        self._init_model_weights()
        # make sure checkpoint-related hooks are triggered after `before_run`
        self.load_or_resume()

        # Initiate inner count of `optim_wrapper`.
        self.optim_wrapper.initialize_count_status(
            self.model,
            self._train_loop.iter,  # type: ignore
            self._train_loop.max_iters)  # type: ignore

        # Maybe compile the model according to options in self.cfg.compile
        # This must be called **AFTER** model has been wrapped.
        self._maybe_compile('train_step')

        model = self.train_loop.run()  # type: ignore
        self.call_hook('after_run')
        return model