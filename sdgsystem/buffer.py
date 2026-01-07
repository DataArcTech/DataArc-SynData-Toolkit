import os
import numpy as np
import threading
from typing import List, Iterable, Any

from .utils import load_json, save_json
from .models import ModelUsageCounter


class TaskBuffer:
    def __init__(self,  
        total: int, 
        save_dir: str, 
        save_step: int = 4
    ) -> None:
        """
        Initialize the buffer.
        Args:
            total: The total number of samples to be processed (completed + remain).
            save_dir: The directory for saving temporary results and the progress.
            save_step: The number of processing steps between each saving.
        """
        self.save_step = save_step
        self.save_dir = save_dir
        self.usage_path = os.path.join(self.save_dir, "usage.json")
        self.result_path = os.path.join(self.save_dir, "result.json")

        self.usage = {
            "total": total, 
            "completed": 0, 
            "token": 0, 
            "time": 0.0, 
        }
        self.detail_progress: np.ndarray[bool] = np.zeros(total, dtype=bool)
        self.detail_completed: int = 0
        self.tmp_add_progress: List[int] = []

        self._lock = threading.RLock()

    def resize_total(self, total: int):
        if total > self.usage["total"]:
            self.detail_progress: np.ndarray[bool] = np.concatenate([self.detail_progress, np.zeros(total - self.usage["total"], dtype=bool)])
        elif total < self.usage["total"]:
            self.detail_progress: np.ndarray[bool] = self.detail_progress[ : total]
        self.usage["total"] = total

    def add_progress(self, idxs: List[int]):
        """
        Add current progress to buffer
        Args:
            idxs: The index of processed sample added after the last save.
        """
        with self._lock:
            if max(idxs) >= self.usage["total"]:
                raise Exception(f"Error occurred when add progress to buffer: {idxs} to be added >= total={self.usage['total']}")
            self.tmp_add_progress.extend(idxs)

    def load(self, usage_counter: ModelUsageCounter = None) -> Iterable[Any]:
        # save the original total from __init__ before loading
        original_total = self.usage["total"]

        # load progress
        try:
            usage: dict = load_json(self.usage_path)
            for k, v in self.usage.items():
                self.usage[k] = usage.get(k, v)
            self.detail_progress = np.array(usage.get("detail_progress", []), dtype=bool)
            self.detail_completed = sum(self.detail_progress)
            if self.detail_completed != self.usage["completed"]:
                raise Exception(f"Error occurred when load buffer: completed={self.usage['completed']} is not equal to completed in detail_progress={self.detail_completed}")
            self.tmp_add_progress = []

            if usage_counter:
                usage_counter.load_from_dict(usage)
        except:
            pass

        # resize detail_progress to match the current total if different from loaded total
        if original_total != self.usage["total"]:
            self.resize_total(original_total)
        # load results
        results: Iterable[Any] = []
        try:
            results: Iterable[Any] = load_json(self.result_path)
        except:
            pass

        return results

    def save(self, results: Iterable[Any], usage_counter: ModelUsageCounter = None):
        with self._lock:
            if usage_counter:
                self.usage["total"] = usage_counter.total
                self.usage["completed"] = usage_counter.completed
                self.usage["token"] = usage_counter.token
                self.usage["time"] = usage_counter.time

            completed, total = self.usage["completed"], self.usage["total"]
            if completed % self.save_step == 0 or completed == total:
                # verify the progress
                self.detail_completed += len(self.tmp_add_progress)
                if self.detail_completed != completed:
                    raise Exception(f"Error occurred when save buffer: completed={completed} is not equal to completed in detail_progress={self.detail_completed}")

                self.detail_progress[self.tmp_add_progress] = True
                self.tmp_add_progress = []  # clear the temporary progress to be added
                usage = {**self.usage, **{"detail_progress": self.detail_progress.tolist()}}

                # save buffer
                save_json(usage, self.usage_path)
                save_json(results, self.result_path)