import weakref
import taichi as ti
from multitask.hooks import HookBase
from logger import TensorboardWriter

@ti.data_oriented
class BaseTrainer:
    def __init__(self, args, config):
        self.logger = config.get_logger(name=config.get_config()["train"]["name"])
        self.random_seed = config.get_config()["train"]["random_seed"]
        self.num_processes = args.num_processes
        self.training = True
        self.iter = 0
        self.max_iter = 10000  # Default training iterations, can be overwrote by args
        self.writer = TensorboardWriter(config.log_dir,
                                        self.logger,
                                        enabled=(not args.no_tensorboard_train))
        self._hooks = []

    def before_train(self):
        for h in self._hooks:
            h.before_train()

    def before_step(self):
        for h in self._hooks:
            h.before_step()

    def after_step(self):
        for h in self._hooks:
            h.after_step()

    def after_train(self):
        for h in self._hooks:
            h.after_train()

    def run_step(self):
        raise NotImplementedError

    def train(self, start_iter, max_iter):

        self.iter = self.start_iter = start_iter
        self.max_iter = max_iter // self.num_processes
        self.logger.info(f"Starting training from iteration {start_iter}, Number of Processes: {self.num_processes}, "
                         f"Max iterations: {self.max_iter}")

        try:
            self.before_train()
            for self.iter in range(start_iter, self.max_iter):
                self.before_step()
                self.run_step()
                self.after_step()
            # self.iter == max_iter can be used by `after_train` to
            # tell whether the training successfully finished or failed
            # due to exceptions.
            self.iter += 1
        except Exception:
            self.logger.exception("Exception during training:")
            raise
        finally:
            self.after_train()

    def register_hooks(self, hooks):
        hooks = [h for h in hooks if h is not None]
        for h in hooks:
            assert isinstance(h, HookBase)
            # To avoid circular reference, hooks and trainer cannot own each other.
            # This normally does not matter, but will cause memory leak if the
            # involved objects contain __del__:
            # See http://engineering.hearsaysocial.com/2013/06/16/circular-references-in-python/
            h.trainer = weakref.proxy(self)
        self._hooks.extend(hooks)