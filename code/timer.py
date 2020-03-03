import time
from tabulate import tabulate


class Timer:
    timers = {}

    def __init__(self):
        self.time_spent = {}
        self.curr_task = None
        self.curr_task_start = None

    def task(self, taskname):
        self.stop()
        self.curr_task = taskname
        self.curr_task_start = time.time()

    def stop(self):
        if self.curr_task:
            self.time_spent[self.curr_task] = self.time_spent.get(self.curr_task, 0) + (time.time() - self.curr_task_start)

    def report(self):
        self.stop()
        tasktimes = sorted([(task, round(time, 2)) for task, time in self.time_spent.items()], key=lambda x: x[1])
        tasktimes.append(("TOTAL", sum([time for task, time in tasktimes])))
        print(tabulate(tasktimes))

    @classmethod
    def get(cls, name):
        if name not in cls.timers:
            cls.timers[name] = Timer()
        t: Timer = cls.timers[name]
        return t
