from mmengine.hooks import Hook
from mmseg.registry import HOOKS
from mmengine.optim import OptimWrapper, OptimWrapperDict

@HOOKS.register_module()
class WeightDecayLoggingHook(Hook):
    def before_train_iter(self, runner, batch_idx, **kwargs):
        if isinstance(runner.optim_wrapper, OptimWrapper):
            optimizer = runner.optim_wrapper.optimizer
        elif isinstance(runner.optim_wrapper, OptimWrapperDict):
            optimizer = runner.optim_wrapper['main_optimizer'].optimizer
        else:
            print("No valid optimizer wrapper found.")


        if self.every_n_train_iters(runner, 50):
            weight_decay = optimizer.param_groups[0]['weight_decay']
            print(f"weight decay : {weight_decay}")
