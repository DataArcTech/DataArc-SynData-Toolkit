"""
Parallel processing module, which encapsulates the runtime logic for parallel processing of iterables, along with time and token statistics.
"""
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterable, Callable, Any

from .models import ModelUsageCounter

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
        results = [None] * len(iterable_inputs)
        total_tasks = len(iterable_inputs)

        if usage_counter:
            usage_counter.set_parallel()
            kwargs["usage_counter"] = usage_counter

        logger.info(f"Starting parallel execution: {total_tasks} tasks with {self.n_workers} workers")

        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            # Submit all tasks at once
            future_to_idx = {
                executor.submit(process_function, inp, **kwargs): idx for idx, inp in enumerate(iterable_inputs)
            }

            logger.info(f"Submitted {len(future_to_idx)} tasks to thread pool")

            st = time.time()
            completed = 0

            try:
                for future in as_completed(future_to_idx, timeout=self.timeout):
                    try:
                        result = future.result()
                        # ensure the order of the results is preserved.
                        index = future_to_idx[future]
                        results[index] = result

                        completed += 1
                        logger.info(f"Task {index} completed ({completed}/{total_tasks})")

                        # estimate the token and time usage
                        if usage_counter:
                            usage_counter.set_parallel_time(time.time() - st)
                            usage_counter.estimate_usage(n=n)
                    except Exception as e:
                        index = future_to_idx[future]
                        logger.error(f"Task {index} failed with error: {e}", exc_info=True)
                        raise e

            except TimeoutError as e:
                logger.error(f"Parallel execution timed out after {self.timeout}s ({completed}/{total_tasks} tasks completed)")
                raise e
            except Exception as e:
                logger.error(f"Parallel execution failed after {completed}/{total_tasks} tasks completed: {e}")
                raise e

        elapsed = time.time() - st
        logger.info(f"Parallel execution completed: {total_tasks} tasks in {elapsed:.2f}s")

        return results