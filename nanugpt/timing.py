# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import gc
import timeit
from types import TracebackType
from typing import Optional, Callable


class Timing:
    """Context manager that measures the time elapsed in a block of code."""

    def __init__(self, name: str, disable_gc: Optional[bool] = False,
                 verbose: Optional[bool] = False,
                 # hooks can be used for torch.cuda.synchronize() or other custom code
                 start_hook:Optional[Callable]=None, end_hook:Optional[Callable]=None) -> None:
        self.name = name
        self.disable_gc = disable_gc
        self.verbose = verbose
        self.start_time = None
        self.end_time = None
        self.counter = 0 # caller can use this to count iterations but not used in this class
        self.start_hook = start_hook
        self.end_hook = end_hook

    def __enter__(self) -> 'Timing':
        self.is_gc_enabled = gc.isenabled()

        if self.disable_gc:
            gc.disable()

        if self.start_hook:
            self.start_hook()
        self.start_time = timeit.default_timer()
        self.counter = 0

        return self

    def reset_start_time(self):
        self.start_time = timeit.default_timer()

    def __exit__(self, exc_type: type[BaseException], exc_val: BaseException, exc_tb: TracebackType) -> None:
        if self.disable_gc and self.is_gc_enabled:
            gc.enable()

        if self.end_hook:
            self.end_hook()

        self.end_time = timeit.default_timer()

        if self.verbose:
            print(f"{self.name}: {self.elapsed:.4g} secs")


    @property
    def elapsed(self) -> float:
        """Return the elapsed time in seconds."""

        return (self.end_time or 0) - (self.start_time or 0)