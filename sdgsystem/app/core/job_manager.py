"""Job managers for handling SDG and training job lifecycle."""
import uuid
import logging
from typing import Optional, Callable, Any
from concurrent.futures import ThreadPoolExecutor

from .progress import SDGProgressReporter, TrainProgressReporter

logger = logging.getLogger(__name__)


class SDGJobManager:
    """Manages SDG job execution."""

    def __init__(self):
        self._current_job: Optional[SDGProgressReporter] = None
        self._current_service: Any = None  # SDGService instance
        self._executor = ThreadPoolExecutor(max_workers=1)

    @property
    def current_job(self) -> Optional[SDGProgressReporter]:
        return self._current_job

    @property
    def current_service(self) -> Any:
        return self._current_service

    def set_service(self, service: Any):
        """Store the SDG service for later refinement. Cleans up previous service."""
        if self._current_service is not None and hasattr(self._current_service, '_cleanup_models'):
            try:
                self._current_service._cleanup_models()
            except Exception as e:
                logger.warning(f"Failed to cleanup previous SDG service models: {e}")
        self._current_service = service

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
        def wrapper():
            try:
                task_fn(reporter, *args, **kwargs)
            except Exception as e:
                reporter.fail(
                    code="execution_error",
                    message=str(e),
                    details={"type": type(e).__name__}
                )

        self._executor.submit(wrapper)

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
