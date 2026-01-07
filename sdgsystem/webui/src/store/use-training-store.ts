import { create } from 'zustand'
import { devtools } from 'zustand/middleware'

export interface TrainingTaskConfig {
  method: string

  // Model 配置
  model: {
    path: string
    enable_gradient_checkpointing: boolean
    use_remove_padding: boolean
    use_liger: boolean
    lora_rank: number
    lora_alpha: number
    target_modules: string
  }

  // Data 配置
  data: {
    prompt_key: string
    response_key: string
    train_batch_size: number
    micro_batch_size_per_gpu: number
    max_length?: number
    truncation?: string
    // GRPO 特定
    max_prompt_length?: number
    max_response_length?: number
    filter_overlong_prompts?: boolean
  }

  // SFT 的 optimizer
  optim: {
    lr: number
  }

  // GRPO 的 actor
  actor?: {
    optim: {
      lr: number
    }
    ppo_mini_batch_size: number
    ppo_micro_batch_size_per_gpu: number
    use_kl_loss?: boolean
    kl_loss_coef?: number
    kl_loss_type?: string
    entropy_coeff?: number
    fsdp_config?: {
      param_offload: boolean
      optimizer_offload: boolean
    }
  }

  // GRPO 新增对象
  rollout?: {
    n: number
    gpu_memory_utilization: number
    tensor_model_parallel_size: number
    mode: string
  }

  ref?: {
    fsdp_config?: {
      param_offload: boolean
    }
  }

  algorithm?: {
    adv_estimator: string
    use_kl_in_reward: boolean
  }

  reward?: {
    data_source?: string
    custom_reward_function?: {
      path: string
      name: string
    }
    reward_model_path?: string
  }

  trainer: {
    project_name: string
    experiment_name: string
    logger: string
    total_epochs: number
    save_freq?: number
    nnodes?: number
    n_gpus_per_node: number
    val_before_train?: boolean
    critic_warmup?: number
    test_freq?: number
    default_local_dir?: string
    wandb_proxy?: string
  }

  // 顶级字段
  ulysses_sequence_parallel_size?: number

  // Wandb 配置
  wandb?: {
    api_key?: string | null
    entity?: string | null
    enable_weave?: boolean
  }
}

interface TrainingState {
  trainingConfig: TrainingTaskConfig
  setTrainingConfig: (config: Partial<TrainingTaskConfig>) => void
}

// SFT 默认配置
export const SFT_DEFAULT_CONFIG: TrainingTaskConfig = {
  method: 'sft',
  model: {
    path: '',
    enable_gradient_checkpointing: true,
    use_remove_padding: true,
    use_liger: false,
    lora_rank: 32,
    lora_alpha: 16,
    target_modules: 'all-linear',
  },
  data: {
    train_batch_size: 32,
    micro_batch_size_per_gpu: 4,
    max_length: 2048,
    truncation: 'right',
    prompt_key: 'input',
    response_key: 'output',
  },
  optim: {
    lr: 1e-5,
  },
  trainer: {
    project_name: 'sdg-sft',
    experiment_name: '',
    logger: 'console,wandb',
    total_epochs: 3,
    save_freq: 100,
    nnodes: 1,
    n_gpus_per_node: 1,
    wandb_proxy: 'http://127.0.0.1:7890',
  },
  ulysses_sequence_parallel_size: 1,
  wandb: {
    api_key: null,
    entity: null,
  },
}

// GRPO 默认配置
export const GRPO_DEFAULT_CONFIG: TrainingTaskConfig = {
  method: 'grpo',
  model: {
    path: 'Qwen/Qwen2.5-0.5B-Instruct',
    enable_gradient_checkpointing: true,
    use_remove_padding: true,
    use_liger: false,
    lora_rank: 32,
    lora_alpha: 16,
    target_modules: 'all-linear',
  },
  data: {
    train_batch_size: 32,
    micro_batch_size_per_gpu: 4,
    max_prompt_length: 512,
    max_response_length: 1024,
    truncation: 'right',
    filter_overlong_prompts: true,
    prompt_key: 'input',
    response_key: 'output',
  },
  optim: {
    lr: 1e-6,
  },
  actor: {
    optim: {
      lr: 1e-6,
    },
    ppo_mini_batch_size: 16,
    ppo_micro_batch_size_per_gpu: 4,
    use_kl_loss: true,
    kl_loss_coef: 0.001,
    kl_loss_type: 'low_var_kl',
    entropy_coeff: 0.0,
    fsdp_config: {
      param_offload: false,
      optimizer_offload: false,
    },
  },
  rollout: {
    n: 4,
    gpu_memory_utilization: 0.4,
    tensor_model_parallel_size: 1,
    mode: 'async',
  },
  ref: {
    fsdp_config: {
      param_offload: true,
    },
  },
  algorithm: {
    adv_estimator: 'grpo',
    use_kl_in_reward: false,
  },
  reward: {
    data_source: '',
    custom_reward_function: {
      path: '',
      name: '',
    },
  },
  trainer: {
    project_name: 'sdg-grpo',
    experiment_name: '',
    logger: 'console,wandb',
    total_epochs: 1,
    save_freq: 100,
    test_freq: -1,
    nnodes: 1,
    n_gpus_per_node: 1,
    val_before_train: false,
    critic_warmup: 0,
    wandb_proxy: 'http://127.0.0.1:7890',
  },
  wandb: {
    api_key: null,
    entity: null,
    enable_weave: false,
  },
}

const DEFAULT_TRAINING_CONFIG: TrainingTaskConfig = SFT_DEFAULT_CONFIG

export const useTrainingStore = create<TrainingState>()(
  devtools(
    set => ({
      trainingConfig: DEFAULT_TRAINING_CONFIG,
      setTrainingConfig: newConfig =>
        set(state => ({
          trainingConfig: { ...state.trainingConfig, ...newConfig },
        })),
    }),
    { name: 'TrainingStore' },
  ),
)
