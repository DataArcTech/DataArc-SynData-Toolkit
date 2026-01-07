from typing import Dict, Callable, Optional
import threading
import logging

logger = logging.getLogger(__name__)


class ModelUsageCounter:
    """Token and Time counter for counting and estimating usage."""
    
    def __init__(self,
        total: int = 0,
        name: str = "Model",
        parallel: bool = False,
        on_update: Optional[Callable[["ModelUsageCounter"], None]] = None
    ) -> None:
        """
        Initialize token and time counter

        Args:
            total: Anticipated number of each iteration unit.
            name: The module name used for identification when using logging.info:
                [{name} Usage]: completed=XXXX, token=XXXX, time=XXXX || remain=XXXX, remain_token_anticipation=XXXX, remain_time_anticipation=XXXX
            parallel: Whether to be used in parallel processing. If parallel, time needs to be counted additionally.
            on_update: Optional callback called after estimate_usage with self as argument.
        """
        self.name = name
        self.total = total

        self.token: int = 0
        self.time: float = 0        # measured in seconds

        self.completed: int = 0    # number of iteration unit
        self.remain: int = self.total - self.completed     # number of remain iteration unit

        # parallel setting
        self._is_parallel = parallel
        self._lock = threading.RLock()

        # callback for progress updates
        self._on_update = on_update

    def load_from_dict(self, usage: Dict):
        self.total = usage.get("total", 0)
        self.name = usage.get("name", self.name)
        self.token = usage.get("token", 0)
        self.time = usage.get("time", 0.0)
        self.completed = usage.get("completed", 0)
        self.remain = self.total - self.completed
        if self.remain < 0:    # verify the remain to > 0
            self.remain = 0
            self.total = self.completed

    def add_usage(self, 
        n_token: int, 
        time: float,
    ):
        """
        add token and time usage to this counter.
        Args: 
            n_token: token cost
            time: time cost
        
        if add_usage in parallel processing, time will not be counted. 
        Please use set_parallel_time(time) in parallel processing.
        """
        with self._lock:
            self.token += n_token
            if not self._is_parallel:
                self.time += time

    def set_sequential(self):
        self._is_parallel = False

    def set_parallel(self):
        self._is_parallel = True

    def set_parallel_time(self, time: float):
        if self._is_parallel:
            self.time = time

    def set_on_update(self, callback: Optional[Callable[["ModelUsageCounter"], None]]):
        """Set callback to be called after estimate_usage."""
        self._on_update = callback

    def resize_total(self, total: int):
        """Resize total and recalculate remain accordingly."""
        self.total = total
        self.remain = self.total - self.completed
        if self.remain < 0:
            self.remain = 0

    def estimate_usage(self, n):
        """
        estimate the token and time usage of n iteration units
        Args:
            n: the number of iteration units
        """
        with self._lock:
            n = min(self.remain, n)
            self.remain -= n
            self.completed += n

            avg_token = self.token / self.completed
            avg_time = self.time / self.completed

            remain_token_anticipation = int(avg_token * self.remain)
            remain_time_anticipation = avg_time * self.remain

            logger.info(f"[{self.name} Usage]: completed={self.completed}, token={self.token}, time={self.time:.2f} || remain={self.remain}, remain_token_anticipation={remain_token_anticipation}, remain_time_anticipatioin={remain_time_anticipation:.2f}")

        # call callback outside of lock to avoid deadlock
        if self._on_update:
            self._on_update(self)

    @property
    def estimated_remaining_tokens(self) -> int:
        """Estimate remaining tokens based on average usage."""
        if self.completed == 0:
            return 0
        avg_token = self.token / self.completed
        return int(avg_token * self.remain)

    @property
    def estimated_remaining_time(self) -> float:
        """Estimate remaining time based on average usage."""
        if self.completed == 0:
            return 0.0
        avg_time = self.time / self.completed
        return avg_time * self.remain

    def __str__(self) -> str:
        return f"[{self.name} Usage]: completed={self.completed}, remain={self.remain}, token={self.token}, time={self.time:.2f}"
