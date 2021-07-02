import os
import time
from util import MetricTracker


class HookBase:
    def __init__(self):
        self.trainer = None

    def before_train(self):
        pass

    def before_step(self):
        pass

    def after_step(self):
        pass

    def after_train(self):
        pass


class Checkpointer(HookBase):
    def __init__(self, save_path):
        super(Checkpointer, self).__init__()
        self.save_path = save_path

    def before_train(self):
        os.makedirs(self.save_path, exist_ok=True)

    def after_step(self):
        pass

    def after_train(self):
        pass


class InfoPrinter(HookBase):
    def __init__(self):
        super(InfoPrinter, self).__init__()


class Timer(HookBase):
    def __init__(self):
        super(Timer, self).__init__()
        self.train_start = 0.0
        self.train_end = 0.0
        self.step_start = 0.0
        self.step_end = 0.0

    def before_train(self):
        self.train_start = time.time()

    def before_step(self):
        self.step_start = time.time()

    def after_step(self):
        self.step_end = time.time()

    def after_train(self):
        self.train_end = time.time()


class MetricWriter(HookBase):
    def __init__(self):
        super(MetricWriter, self).__init__()
        self.writer = None
        self.train_metrics = None
        self.valid_metrics = None

    def reset(self):
        if self.writer is None:
            self.writer = self.trainer.writer
            self.train_metrics = MetricTracker(*[m for m in self.trainer.metrics_train], writer=self.writer)
            self.valid_metrics = MetricTracker(*[m for m in self.trainer.metrics_validation], writer=self.writer)
        self.train_metrics.reset()

    def before_train(self):
        self.reset()







