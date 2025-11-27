from .base import *
from .local import *
from .web import *
from .distill import *


class TaskExecutor:
    def __init__(self, 
        # config: BaseTaskConfig, 
        config: SDGSTaskConfig, 
        llm: ModelClient
    ) -> None:
        self.config = config
        # self.executors: List[BaseTaskExecutor] = TaskExecutor.get_executors(llm, self.config)
        self.executor: BaseTaskExecutor = TaskExecutor.get_executors(llm, self.config)

    @staticmethod
    def get_executors(llm: ModelClient, config: SDGSTaskConfig) -> List[BaseTaskExecutor]:
        # executors: List[BaseTaskExecutor] = []
        
        # sub_config = config.local_task_config
        # if sub_config:
        #     executors.append(LocalTaskExecutor(sub_config, llm))

        # sub_config = config.web_task_config
        # if sub_config:
        #     executors.append(WebTaskConfig(sub_config, llm))

        # sub_config = config.distill_task_config
        # if sub_config:
        #     executors.append(DistillTaskConfig(sub_config, llm))

        # return executors
        executor: BaseTaskExecutor = None

        if config.task_type == "local":
            sub_config = config.local_task_config
            executor = LocalTaskExecutor(sub_config, llm)

        elif config.task_type == "web":
            sub_config = config.web_task_config
            executor = WebTaskExecutor(sub_config, llm)

        elif config.task_type == "distill":
            sub_config = config.distill_task_config
            executor = DistillTaskExecutor(sub_config, llm)

        else:
            raise Exception(f"no task type of {config.task_type}")

        return executor

    def execute(self, parallel_executor=None) -> Dataset:
        dataset = Dataset()
        # for executor in self.executors:
        #     dataset.extend(executor.execute(parallel_executor=parallel_executor))
        dataset.extend(self.executor.execute(parallel_executor=parallel_executor))

        return dataset