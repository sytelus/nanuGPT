# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# adapted from https://github.com/ildoonet/pystopwatch2/blob/master/pystopwatch2/watch.py

"""
## Basic use:
```
    sw = StopWatch()
    sw.start()
    sw.pause()
    sw.elapsed_total()
```

## Break down timing by sections:
```
    sw.start('section1')
    sw.pause('section1')
    sw.start('section1') # continue adding time for section1
    sw.pause('section1')

    # get timing details for section1
    sw.elapsed_total('section1')
    sw.elapsed_mean('section1')
    sw.elapsed_min('section1')
    sw.elapsed_max('section1')
    sw.elapsed_stddev('section1')
    sw.elapsed_len('section1')

    # report
    sw.report('section1') # returns a dict
    sw.report_all() # returns a dict of dicts
```
## others
```
    sw.enable(False) # ignore any calls for start/pause
    sw.clear('section1') # delete the timing details for section1
    sw.clear_all() # delete all timing details gathered so far
    sw.enable_all(False) # ignore any calls for start/pause for all sections
```

"""


from typing import Optional
import threading
import timeit
from collections import defaultdict
from enum import Enum


class _ClockState(Enum):
    STOPPED = 0
    RUN = 1

class _Clock:
    tag_default = '__default1958__'
    th_lock = threading.Lock()

    def __init__(self, enabled=True):
        self._prev_time = timeit.default_timer()
        self._times = []
        self.state = _ClockState.STOPPED
        self.enabled = enabled

    def start(self):
        if not self.enabled:
            return
        if self.state == _ClockState.RUN:
            return
        self.state = _ClockState.RUN
        self._prev_time = timeit.default_timer()

    def pause(self):
        if not self.enabled:
            return
        if self.state == _ClockState.STOPPED:
            return
        delta = timeit.default_timer() - self._prev_time
        self._times.append(delta)
        self.state = _ClockState.STOPPED

    def elapsed_total(self):
        return sum(self._times)

    def elapsed_mean(self):
        return sum(self._times) / len(self._times)

    def elapsed_min(self):
        return min(self._times)

    def elapsed_max(self):
        return max(self._times)

    def elapsed_stddev(self):
        avg = self.elapsed_mean()
        return (sum([(t - avg)**2 for t in self._times]) / len(self._times))**0.5

    def __len__(self):
        return len(self._times)

    def __str__(self):
        return 'state=%s elapsed=%.4f prev_time=%.8f' % (self.state, self.elapsed_total(), self._prev_time)

    def __repr__(self):
        return self.__str__()


class StopWatch:
    stopwatch:Optional['StopWatch'] = None

    def __init__(self):
        self.clocks = defaultdict(lambda: _Clock())

    def start(self, tag=None):
        if tag is None:
            tag = _Clock.tag_default
        with _Clock.th_lock:
            self.clocks[tag].start()

    def pause(self, tag=None):
        if tag is None:
            tag = _Clock.tag_default
        with _Clock.th_lock:
            self.clocks[tag].pause()

    def stop(self, tag=None):
        self.pause(tag)

    def clear(self, tag=None):
        if tag is None:
            tag = _Clock.tag_default
        del self.clocks[tag]

    def clear_all(self):
        self.clocks.clear()

    def enable(self, enabled:bool, tag=None):
        if tag is None:
            tag = _Clock.tag_default
        self.clocks[tag].enabled = enabled

    def enable_all(self, enabled:bool):
        for c in self.clocks.values():
            c.enabled = enabled

    def elapsed_total(self, tag=None):
        if tag is None:
            tag = _Clock.tag_default
        return self.clocks[tag].elapsed_total()

    def elapsed_mean(self, tag=None):
        if tag is None:
            tag = _Clock.tag_default
        return self.clocks[tag].elapsed_mean()

    def elapsed_min(self, tag=None):
        if tag is None:
            tag = _Clock.tag_default
        return self.clocks[tag].elapsed_min()

    def elapsed_max(self, tag=None):
        if tag is None:
            tag = _Clock.tag_default
        return self.clocks[tag].elapsed_max()

    def elapsed_stddev(self, tag=None):
        if tag is None:
            tag = _Clock.tag_default
        return self.clocks[tag].elapsed_stddev()

    def elapsed_len(self, tag=None):
        if tag is None:
            tag = _Clock.tag_default
        return len(self.clocks[tag])

    def __len__(self):
        return len(self.clocks)

    def keys(self):
        return self.clocks.keys()

    def __str__(self):
        return '\n'.join(['%s: %s' % (k, v) for k, v in self.clocks.items()])

    def __repr__(self):
        return self.__str__()

    def report(self, tag=None)->dict:
        if tag is None:
            tag = _Clock.tag_default
        return {
            'elapsed_total': self.elapsed_total(tag),
            'elapsed_mean': self.elapsed_mean(tag),
            'elapsed_min': self.elapsed_min(tag),
            'elapsed_max': self.elapsed_max(tag),
            'elapsed_stddev': self.elapsed_stddev(tag),
            'elapsed_len': self.elapsed_len(tag),
        }
    def report_all(self)->dict:
        return {k:self.report(k) for k in self.keys()}

    @staticmethod
    def set(instance:'StopWatch')->None:
        StopWatch.stopwatch = instance

    @staticmethod
    def get()->Optional['StopWatch']:
        return StopWatch.stopwatch