"""Job managers for handling SDG and training job lifecycle."""
import uuid
import logging
import threading
from typing import Optional, Callable, Any, List
from concurrent.futures import ThreadPoolExecutor

from .progress import SDGProgressReporter, TrainProgressReporter

logger = logging.getLogger(__name__)


class SDGJobManager:
    """Manages SDG job execution."""

    def __init__(self):
        self._current_job: Optional[SDGProgressReporter] = None
        self._current_service: Any = None  # SDGService instance
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._lock = threading.Lock()
        # Services that have a task submitted to the executor which has not finished yet
        self._busy_services: List[Any] = []
        # Replaced services whose model release must wait until their task finishes
        self._stale_services: List[Any] = []

    @property
    def current_job(self) -> Optional[SDGProgressReporter]:
        return self._current_job

    @property
    def current_service(self) -> Any:
        return self._current_service

    def set_service(self, service: Any):
        """Store the SDG service for later phases and release the previous one.

        Models of the previous service are released immediately unless one of its
        tasks is still queued or running on the worker thread. In that case the
        release is deferred to the worker (see run_job) so models are never torn
        down while a job is still using them.
        """
        with self._lock:
            previous = self._current_service
            self._current_service = service
            if previous is None or previous is service:
                return
            if self._is_listed(previous, self._busy_services):
                self._stale_services.append(previous)
                return
        self._release_models(previous)

    def create_job(self, task_type: str, task_name: str) -> SDGProgressReporter:
        """Create a new job."""
        job_id = str(uuid.uuid4())
        self._current_job = SDGProgressReporter(
            job_id=job_id,
            task_type=task_type,
            task_name=task_name
        )
        return self._current_job

    def get_job(self, job_id: str) -> Optional[SDGProgressReporter]:
        """Get job by ID."""
        if self._current_job and self._current_job.job_id == job_id:
            return self._current_job
        return None

    def run_job(self, reporter: SDGProgressReporter, task_fn: Callable, *args, **kwargs):
        """Run job task in background thread."""
        with self._lock:
            service = self._current_service
            self._busy_services.append(service)

        def wrapper():
            try:
                task_fn(reporter, *args, **kwargs)
            except Exception as e:
                reporter.fail(
                    code="execution_error",
                    message=str(e),
                    details={"type": type(e).__name__}
                )
            finally:
                self._on_task_finished(service)

        self._executor.submit(wrapper)

    def _on_task_finished(self, service: Any):
        """Release a replaced service's models once none of its tasks are pending."""
        with self._lock:
            self._remove_one(self._busy_services, service)
            release = (
                self._is_listed(service, self._stale_services)
                and not self._is_listed(service, self._busy_services)
            )
            if release:
                self._stale_services = [s for s in self._stale_services if s is not service]
        if release:
            self._release_models(service)

    @staticmethod
    def _is_listed(target: Any, items: List[Any]) -> bool:
        return any(item is target for item in items)

    @staticmethod
    def _remove_one(items: List[Any], target: Any) -> None:
        for index, item in enumerate(items):
            if item is target:
                del items[index]
                return

    @staticmethod
    def _release_models(service: Any) -> None:
        cleanup = getattr(service, "_cleanup_models", None)
        if cleanup is None:
            return
        try:
            cleanup()
        except Exception as e:
            logger.warning(f"Failed to release models of replaced SDG service: {e}")

    def cancel_job(self, job_id: str) -> bool:
        """Cancel current job."""
        if self._current_job and self._current_job.job_id == job_id:
            self._current_job.cancel()
            return True
        return False


class TrainJobManager:
    """Manages training job execution."""

    def __init__(self):
        self._current_job: Optional[TrainProgressReporter] = None
        self._current_service: Any = None
        self._executor = ThreadPoolExecutor(max_workers=1)

    @property
    def current_job(self) -> Optional[TrainProgressReporter]:
        return self._current_job

    @property
    def current_service(self) -> Any:
        return self._current_service

    def set_service(self, service: Any):
        """Store the training service."""
        self._current_service = service

    def create_job(self, method: str, config: dict) -> TrainProgressReporter:
        """Create a new training job."""
        job_id = str(uuid.uuid4())
        self._current_job = TrainProgressReporter(
            job_id=job_id,
            method=method,
            config=config
        )
        return self._current_job

    def get_job(self, job_id: str) -> Optional[TrainProgressReporter]:
        """Get job by ID."""
        if self._current_job and self._current_job.job_id == job_id:
            return self._current_job
        return None

    def run_job(self, reporter: TrainProgressReporter, training_service: Any):
        """Run training job in background thread."""
        def wrapper():
            try:
                reporter.set_running()

                method = training_service.method
                if method == "sft":
                    return_code = training_service.run_sft(
                        log_callback=reporter.add_log
                    )
                elif method == "grpo":
                    return_code = training_service.run_grpo(
                        log_callback=reporter.add_log
                    )
                else:
                    reporter.fail(
                        code="unsupported_method",
                        message=f"Method '{method}' not implemented"
                    )
                    return

                if return_code == 0:
                    reporter.complete()
                else:
                    reporter.fail(
                        code="training_failed",
                        message=f"Training exited with code {return_code}"
                    )

            except Exception as e:
                reporter.fail(code="execution_error", message=str(e))

        self._executor.submit(wrapper)

    def cancel_job(self, job_id: str) -> bool:
        """Cancel current training job and terminate the subprocess."""
        if self._current_job and self._current_job.job_id == job_id:
            # Cancel the training service (terminates subprocess)
            if self._current_service:
                self._current_service.cancel()
            # Update job status
            self._current_job.cancel()
            return True
        return False


# Global instances
sdg_job_manager = SDGJobManager()
train_job_manager = TrainJobManager()
