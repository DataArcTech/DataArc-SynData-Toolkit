import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterable, Callable, Any

from .models import ModelUsageCounter
from .buffer import TaskBuffer

logger = logging.getLogger(__name__)


class ParallelExecutor:
    def __init__(self, 
        n_workers: int = 4, 
        timeout: float = 300
    ) -> None:
        """
        Initialize parallel executor

        Args:
            n_workers: number of workers for parallel processing
            timeout: limit of time for each process
        """
        self.n_workers = n_workers
        self.timeout = timeout

    def execute(
        self,
        iterable_inputs: Iterable[Any],
        process_function: Callable[[Any], Any],
        usage_counter: ModelUsageCounter = None,
        n: int = 1,
        buffer: TaskBuffer = None,
        **kwargs
    ) -> Any:
        """
        Execute the parallel processing

        Args:
            iterable_inputs: iterable inputs to be processed
            process_function: function to process each item of iterable_inputs
            usage_counter: instance to count and estimate the token and time usage of model
            n: number of samples advanced each time usage is estimated, especially when iterable_inputs are batched.
                e.g. iterable_inputs = [[1, 2, 3], [4, 5, 6], ..., [22, 23]], then n=3
            **kwargs: additional arguments for process_function, which are fixed for each iteration
        """
        # load buffer
        if buffer:
            results: list = buffer.load(usage_counter)
            remain = len(iterable_inputs) - len(results)
            if remain < 0:
                raise Exception(f"Number of iterable inputs ({len(iterable_inputs)}) is less than the number of results in buffer ({len(results)})!")
            elif remain > 0:
                results = results + [None] * remain
        else:
            results = [None] * len(iterable_inputs)

        # set usage counter
        if usage_counter:
            usage_counter.set_parallel()
            kwargs["usage_counter"] = usage_counter

        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            future_to_idx = {
                executor.submit(process_function, inp, **kwargs): idx for idx, inp in enumerate(iterable_inputs) if not buffer or not buffer.detail_progress[idx]
            }   # filter out processed samples (process all if no buffer)

            st = time.time()
            try:
                for future in as_completed(future_to_idx, timeout=self.timeout):
                    try:
                        result = future.result()
                        # ensure the order of the results is preserved.
                        index = future_to_idx[future]
                        results[index] = result

                        # estimate the token and time usage
                        if usage_counter:
                            usage_counter.set_parallel_time(time.time() - st)
                            usage_counter.estimate_usage(n=n)

                        if buffer:
                            buffer.add_progress([index])
                            buffer.save(results, usage_counter)

                    except Exception as e:
                        raise e

            except Exception as e:
                raise e

        return results