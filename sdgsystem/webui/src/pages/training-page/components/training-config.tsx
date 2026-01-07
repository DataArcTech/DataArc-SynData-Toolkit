import { MinusOutlined, PlusOutlined, QuestionCircleOutlined } from '@ant-design/icons'
import {
  Button,
  Card,
  Collapse,
  ConfigProvider,
  Divider,
  Form,
  Input,
  InputNumber,
  message,
  Select,
  Switch,
  Tooltip,
} from 'antd'
import { useEffect, useRef, useState } from 'react'
import GrpoIcon from '@/assets/icon/traning/grpo.svg?react'
import SftIcon from '@/assets/icon/traning/sft.svg?react'
import { CustomUpload } from '@/components/custom-upload'
import SvgIcon from '@/components/svg-icon'
import { useAntdTheme } from '@/hooks/use-antd-theme'
import {
  GRPO_DEFAULT_CONFIG,
  SFT_DEFAULT_CONFIG,
  useTrainingStore,
} from '@/store/use-training-store'

type TrainingMethod = 'sft' | 'grpo'

interface TrainingConfigProps {
  onStartTraining?: (jobId: string) => void
}

interface NumberStepperProps {
  value?: number
  onChange?: (value: number | null) => void
  min?: number
  max?: number
  step?: number
  style?: React.CSSProperties
}

const NumberStepper = ({ value, onChange, min = 0, max, step = 1, style }: NumberStepperProps) => {
  const token = useAntdTheme()

  const handleDecrease = () => {
    const newValue = (value ?? 0) - step
    if (newValue >= min) {
      onChange?.(newValue)
    }
  }

  const handleIncrease = () => {
    const newValue = (value ?? 0) + step
    if (max === undefined || newValue <= max) {
      onChange?.(newValue)
    }
  }

  return (
    <div
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        background: token.colorFillTertiary,
        borderRadius: token.borderRadius,
        gap: 8,
        width: 'fit-content',
        ...style,
      }}
    >
      <Button
        type="text"
        icon={<MinusOutlined style={{ fontSize: 14 }} />}
        onClick={handleDecrease}
        disabled={(value ?? 0) <= min}
        style={{
          width: 24,
          height: 24,
          padding: 0,
          background: 'transparent',
          border: 'none',
        }}
      />
      <div
        style={{
          background: token.colorBgContainer,
          border: `1px solid ${token.colorBorder}`,
          borderRadius: token.borderRadius,
          padding: '0 4px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          height: 32,
        }}
      >
        <InputNumber
          value={value}
          onChange={onChange}
          min={min}
          max={max}
          step={step}
          controls={false}
          variant="borderless"
          style={{
            width: 48,
            textAlign: 'center',
            padding: 0,
          }}
          styles={{
            input: {
              textAlign: 'center',
              padding: 0,
            },
          }}
        />
      </div>
      <Button
        type="text"
        icon={<PlusOutlined style={{ fontSize: 14 }} />}
        onClick={handleIncrease}
        disabled={max !== undefined && (value ?? 0) >= max}
        style={{
          width: 24,
          height: 24,
          padding: 0,
          background: 'transparent',
          border: 'none',
        }}
      />
    </div>
  )
}

export default function TrainingConfig({ onStartTraining }: TrainingConfigProps) {
  const token = useAntdTheme()
  const [form] = Form.useForm()
  const [submitting, setSubmitting] = useState(false)
  const [hoveredMethod, setHoveredMethod] = useState<TrainingMethod | null>(null)

  const { trainingConfig, setTrainingConfig } = useTrainingStore()
  const trainingMethod = (trainingConfig.method as TrainingMethod) || 'sft'
  const isSettingFormValuesRef = useRef(false)
  const rewardMode = Form.useWatch('rewardMode', form)
  const trainingMethodRef = useRef<TrainingMethod>('sft')

  const renderLabelWithTip = (label: string, tooltip: string, _required?: boolean) => (
    <span className="inline-flex items-center gap-1">
      <span>{label}</span>
      <Tooltip title={tooltip}>
        <QuestionCircleOutlined style={{ color: token.colorTextTertiary }} />
      </Tooltip>
    </span>
  )

  // Initialize form from store
  // biome-ignore lint/correctness/useExhaustiveDependencies: Initialize form only on mount or method change
  useEffect(() => {
    isSettingFormValuesRef.current = true
    trainingMethodRef.current = trainingMethod
    const values = {
      taskName: trainingConfig.trainer.experiment_name,
      baseModelPath: trainingConfig.model.path,
      wandbToken: trainingConfig.wandb?.api_key,
      learningRate: trainingConfig.optim.lr,

      // Model fields
      useRemovePadding: trainingConfig.model.use_remove_padding ?? true,
      useLiger: trainingConfig.model.use_liger ?? false,
      loraAlpha: trainingConfig.model.lora_alpha ?? 16,
      targetModules: trainingConfig.model.target_modules ?? 'all-linear',

      // Shared / Unified
      totalEpochs: trainingConfig.trainer.total_epochs,
      trainBatchSize: trainingConfig.data.train_batch_size,

      // Data keys
      promptKey: trainingConfig.data.prompt_key,
      responseKey: trainingConfig.data.response_key,

      // Advanced / Optional
      maxLength: trainingConfig.data.max_length,
      truncation: trainingConfig.data.truncation,
      microBatchSize: trainingConfig.data.micro_batch_size_per_gpu,
      numNodes: trainingConfig.trainer.nnodes,
      gpusPerNode: trainingConfig.trainer.n_gpus_per_node,

      // LoRA
      useLoRAFineTuning: trainingConfig.model.lora_rank > 0,
      loraRank: trainingConfig.model.lora_rank > 0 ? trainingConfig.model.lora_rank : 8,

      // GRPO specific - read from correct locations
      microBatchSizePerGPU: trainingConfig.data.micro_batch_size_per_gpu,
      rolloutN: trainingConfig.rollout?.n,
      saveFrequency: trainingConfig.trainer.save_freq,
      maxPromptLength: trainingConfig.data.max_prompt_length,
      maxResponseLength: trainingConfig.data.max_response_length,
      testFrequency: trainingConfig.trainer.test_freq,
      rolloutGpuMemoryUtil: trainingConfig.rollout?.gpu_memory_utilization,
      ppoMiniBatchSize: trainingConfig.actor?.ppo_mini_batch_size,
      ppoMicroBatchSize: trainingConfig.actor?.ppo_micro_batch_size_per_gpu,
      useKLLoss: trainingConfig.actor?.use_kl_loss ?? true,
      klLossCoefficient: trainingConfig.actor?.kl_loss_coef,
      rewardMode: trainingMethod === 'grpo' ? 'Rule Based(Custom)' : undefined,
      dataSource: trainingConfig.reward?.data_source,
      customFunctionPath: trainingConfig.reward?.custom_reward_function?.path,
      customFunctionName: trainingConfig.reward?.custom_reward_function?.name,
      rewardModelPath: trainingConfig.reward?.reward_model_path,
    }
    form.setFieldsValue(values)

    // Reset flag after a tick
    setTimeout(() => {
      isSettingFormValuesRef.current = false
    }, 0)
  }, [trainingConfig.method]) // Re-init when method changes or on mount (method is in store)

  useEffect(() => {
    if (isSettingFormValuesRef.current) return
    if (!rewardMode) return
    if (trainingMethodRef.current !== 'grpo') return
    const resetValues: Record<string, string | undefined> = {}
    if (rewardMode === 'Rule Based(Built in)') {
      resetValues.customFunctionPath = undefined
      resetValues.customFunctionName = undefined
      resetValues.rewardModelPath = undefined
    } else if (rewardMode === 'Rule Based(Custom)') {
      resetValues.dataSource = undefined
      resetValues.rewardModelPath = undefined
    } else if (rewardMode === 'Reward Model') {
      resetValues.dataSource = undefined
      resetValues.customFunctionPath = undefined
      resetValues.customFunctionName = undefined
    }
    form.setFieldsValue(resetValues)
  }, [rewardMode, form])

  // Handle form changes
  // biome-ignore lint/suspicious/noExplicitAny: AntD form values
  const handleValuesChange = (changedValues: any, allValues: any) => {
    if (isSettingFormValuesRef.current) return

    // biome-ignore lint/suspicious/noExplicitAny: Dynamic config update
    const updates: any = {}
    let hasUpdates = false

    // Helper to update nested config
    // biome-ignore lint/suspicious/noExplicitAny: Dynamic config update
    const updateSection = (section: string, key: string, value: any) => {
      if (!updates[section])
        // @ts-expect-error - Dynamic config update
        updates[section] = { ...trainingConfig[section as keyof typeof trainingConfig] }
      updates[section][key] = value
      hasUpdates = true
    }

    if (changedValues.taskName !== undefined) {
      updateSection('trainer', 'experiment_name', changedValues.taskName)
    }
    if (changedValues.baseModelPath !== undefined)
      updateSection('model', 'path', changedValues.baseModelPath)

    // Wandb token to wandb.api_key
    if (changedValues.wandbToken !== undefined) {
      setTrainingConfig({
        wandb: {
          ...trainingConfig.wandb,
          api_key: changedValues.wandbToken,
        },
      })
      return
    }

    // Model fields
    if (changedValues.useRemovePadding !== undefined)
      updateSection('model', 'use_remove_padding', changedValues.useRemovePadding)
    if (changedValues.useLiger !== undefined)
      updateSection('model', 'use_liger', changedValues.useLiger)
    if (changedValues.loraAlpha !== undefined)
      updateSection('model', 'lora_alpha', Number(changedValues.loraAlpha))
    if (changedValues.targetModules !== undefined)
      updateSection('model', 'target_modules', changedValues.targetModules)

    if (changedValues.learningRate !== undefined)
      updateSection('optim', 'lr', changedValues.learningRate)

    // Unified Total Epochs
    if (changedValues.totalEpochs !== undefined)
      updateSection('trainer', 'total_epochs', Number(changedValues.totalEpochs))

    if (changedValues.promptKey !== undefined)
      updateSection('data', 'prompt_key', changedValues.promptKey)
    if (changedValues.responseKey !== undefined)
      updateSection('data', 'response_key', changedValues.responseKey)

    if (changedValues.maxLength !== undefined)
      updateSection('data', 'max_length', Number(changedValues.maxLength))
    if (changedValues.truncation !== undefined)
      updateSection('data', 'truncation', changedValues.truncation)

    if (changedValues.microBatchSize !== undefined)
      updateSection('data', 'micro_batch_size_per_gpu', Number(changedValues.microBatchSize))
    if (changedValues.trainBatchSize !== undefined)
      updateSection('data', 'train_batch_size', Number(changedValues.trainBatchSize))

    // Use standard field names for trainer
    if (changedValues.numNodes !== undefined) {
      setTrainingConfig({
        trainer: { ...trainingConfig.trainer, nnodes: Number(changedValues.numNodes) },
      })
      return
    }
    if (changedValues.gpusPerNode !== undefined)
      updateSection('trainer', 'n_gpus_per_node', Number(changedValues.gpusPerNode))

    // LoRA logic
    if (changedValues.useLoRAFineTuning !== undefined || changedValues.loraRank !== undefined) {
      const useLoRA = allValues.useLoRAFineTuning
      const rank = Number(allValues.loraRank) || 8
      updateSection('model', 'lora_rank', useLoRA ? rank : 0)
    }

    // GRPO fields to correct objects
    if (changedValues.microBatchSizePerGPU !== undefined)
      updateSection('data', 'micro_batch_size_per_gpu', changedValues.microBatchSizePerGPU)
    if (changedValues.maxPromptLength !== undefined)
      updateSection('data', 'max_prompt_length', Number(changedValues.maxPromptLength))
    if (changedValues.maxResponseLength !== undefined)
      updateSection('data', 'max_response_length', Number(changedValues.maxResponseLength))

    // Trainer fields with standard names
    if (changedValues.saveFrequency !== undefined) {
      setTrainingConfig({
        trainer: { ...trainingConfig.trainer, save_freq: Number(changedValues.saveFrequency) },
      })
      return
    }
    if (changedValues.testFrequency !== undefined) {
      setTrainingConfig({
        trainer: { ...trainingConfig.trainer, test_freq: Number(changedValues.testFrequency) },
      })
      return
    }

    // Actor fields - use correct spread order to ensure new values override old ones
    if (changedValues.ppoMiniBatchSize !== undefined) {
      setTrainingConfig({
        actor: {
          ...trainingConfig.actor,
          optim: { lr: trainingConfig.actor?.optim?.lr || 1e-6 },
          ppo_mini_batch_size: Number(changedValues.ppoMiniBatchSize),
          ppo_micro_batch_size_per_gpu: trainingConfig.actor?.ppo_micro_batch_size_per_gpu || 8,
        },
      })
      return
    }
    if (changedValues.ppoMicroBatchSize !== undefined) {
      setTrainingConfig({
        actor: {
          ...trainingConfig.actor,
          optim: { lr: trainingConfig.actor?.optim?.lr || 1e-6 },
          ppo_mini_batch_size: trainingConfig.actor?.ppo_mini_batch_size || 64,
          ppo_micro_batch_size_per_gpu: Number(changedValues.ppoMicroBatchSize),
        },
      })
      return
    }
    if (changedValues.useKLLoss !== undefined) {
      setTrainingConfig({
        actor: {
          ...trainingConfig.actor,
          optim: { lr: trainingConfig.actor?.optim?.lr || 1e-6 },
          ppo_mini_batch_size: trainingConfig.actor?.ppo_mini_batch_size || 64,
          ppo_micro_batch_size_per_gpu: trainingConfig.actor?.ppo_micro_batch_size_per_gpu || 8,
          use_kl_loss: changedValues.useKLLoss,
        },
      })
      return
    }
    if (changedValues.klLossCoefficient !== undefined) {
      setTrainingConfig({
        actor: {
          ...trainingConfig.actor,
          optim: { lr: trainingConfig.actor?.optim?.lr || 1e-6 },
          ppo_mini_batch_size: trainingConfig.actor?.ppo_mini_batch_size || 64,
          ppo_micro_batch_size_per_gpu: trainingConfig.actor?.ppo_micro_batch_size_per_gpu || 8,
          kl_loss_coef: Number(changedValues.klLossCoefficient),
        },
      })
      return
    }

    // Rollout fields
    if (changedValues.rolloutN !== undefined) {
      setTrainingConfig({
        rollout: {
          n: Number(changedValues.rolloutN),
          gpu_memory_utilization: trainingConfig.rollout?.gpu_memory_utilization || 0.6,
          tensor_model_parallel_size: trainingConfig.rollout?.tensor_model_parallel_size || 1,
          mode: trainingConfig.rollout?.mode || 'async',
        },
      })
      return
    }
    if (changedValues.rolloutGpuMemoryUtil !== undefined) {
      setTrainingConfig({
        rollout: {
          n: trainingConfig.rollout?.n || 5,
          gpu_memory_utilization: Number(changedValues.rolloutGpuMemoryUtil),
          tensor_model_parallel_size: trainingConfig.rollout?.tensor_model_parallel_size || 1,
          mode: trainingConfig.rollout?.mode || 'async',
        },
      })
      return
    }

    // Reward fields
    if (changedValues.dataSource !== undefined) {
      setTrainingConfig({
        reward: { ...trainingConfig.reward, data_source: changedValues.dataSource },
      })
      return
    }
    if (changedValues.customFunctionPath !== undefined) {
      setTrainingConfig({
        reward: {
          ...trainingConfig.reward,
          custom_reward_function: {
            path: changedValues.customFunctionPath,
            name: trainingConfig.reward?.custom_reward_function?.name || 'compute_score',
          },
        },
      })
      return
    }
    if (changedValues.customFunctionName !== undefined) {
      setTrainingConfig({
        reward: {
          ...trainingConfig.reward,
          custom_reward_function: {
            path: trainingConfig.reward?.custom_reward_function?.path || '/path/to/reward_fn.py',
            name: changedValues.customFunctionName,
          },
        },
      })
      return
    }
    if (changedValues.rewardModelPath !== undefined) {
      setTrainingConfig({
        reward: { ...trainingConfig.reward, reward_model_path: changedValues.rewardModelPath },
      })
      return
    }

    // Apply accumulated updates
    if (hasUpdates) {
      setTrainingConfig(updates)
    }
  }

  const setTrainingMethod = (method: TrainingMethod) => {
    if (method === 'sft') {
      // Switch to SFT: keep shared fields, add SFT defaults
      setTrainingConfig({
        ...SFT_DEFAULT_CONFIG,
        model: { ...SFT_DEFAULT_CONFIG.model, path: trainingConfig.model.path },
        trainer: {
          ...SFT_DEFAULT_CONFIG.trainer,
          experiment_name: trainingConfig.trainer.experiment_name,
          project_name: trainingConfig.trainer.project_name,
        },
        data: {
          ...SFT_DEFAULT_CONFIG.data,
          prompt_key: trainingConfig.data.prompt_key,
          response_key: trainingConfig.data.response_key,
        },
        wandb: trainingConfig.wandb,
      })
    } else if (method === 'grpo') {
      // Switch to GRPO: keep shared fields, add GRPO defaults
      setTrainingConfig({
        ...GRPO_DEFAULT_CONFIG,
        model: { ...GRPO_DEFAULT_CONFIG.model, path: trainingConfig.model.path },
        trainer: {
          ...GRPO_DEFAULT_CONFIG.trainer,
          experiment_name: trainingConfig.trainer.experiment_name,
          project_name: trainingConfig.trainer.project_name,
        },
        data: {
          ...GRPO_DEFAULT_CONFIG.data,
          prompt_key: trainingConfig.data.prompt_key,
          response_key: trainingConfig.data.response_key,
        },
        wandb: trainingConfig.wandb,
      })
    }
  }

  const handleSaveAndContinue = async () => {
    try {
      setSubmitting(true)
      const values = await form.validateFields()

      // Build API config according to docs/train-form.md standard format
      const config = trainingConfig
      const wandbToken = config.wandb?.api_key

      const hasWandbToken = typeof wandbToken === 'string' && wandbToken.trim().length > 0

      // Base API config structure
      const apiConfig: any = {
        method: config.method,
        model: {
          path: config.model.path,
          enable_gradient_checkpointing: config.model.enable_gradient_checkpointing,
          use_remove_padding: config.model.use_remove_padding,
          use_liger: config.model.use_liger,
          lora_rank: config.model.lora_rank,
          lora_alpha: config.model.lora_alpha,
          target_modules: config.model.target_modules,
        },
        data: {
          prompt_key: config.data.prompt_key,
          response_key: config.data.response_key,
          train_batch_size: config.data.train_batch_size,
          micro_batch_size_per_gpu: config.data.micro_batch_size_per_gpu,
        },
        trainer: {
          project_name: config.trainer.project_name,
          experiment_name: config.trainer.experiment_name,
          logger: hasWandbToken ? config.trainer.logger : 'console',
          total_epochs: config.trainer.total_epochs,
          n_gpus_per_node: config.trainer.n_gpus_per_node,
          ...(config.trainer.wandb_proxy && { wandb_proxy: config.trainer.wandb_proxy }),
        },
        ulysses_sequence_parallel_size: config.ulysses_sequence_parallel_size || 1,
      }

      // SFT specific fields
      if (config.method === 'sft') {
        apiConfig.data.max_length = config.data.max_length
        apiConfig.data.truncation = config.data.truncation
        apiConfig.optim = {
          lr: config.optim.lr,
        }
        if (config.trainer.save_freq) apiConfig.trainer.save_freq = config.trainer.save_freq
        if (config.trainer.nnodes) apiConfig.trainer.nnodes = config.trainer.nnodes
      }

      // GRPO specific fields
      if (config.method === 'grpo') {
        apiConfig.data.max_prompt_length = config.data.max_prompt_length
        apiConfig.data.max_response_length = config.data.max_response_length
        apiConfig.data.filter_overlong_prompts = config.data.filter_overlong_prompts ?? true

        if (config.actor) apiConfig.actor = config.actor
        if (config.rollout) apiConfig.rollout = config.rollout
        if (config.ref) apiConfig.ref = config.ref
        if (config.algorithm) apiConfig.algorithm = config.algorithm

        // Handle reward config based on reward mode
        const rewardMode = values.rewardMode
        if (rewardMode === 'Rule Based(Built in)' && values.dataSource) {
          apiConfig.reward = {
            data_source: values.dataSource,
          }
        } else if (rewardMode === 'Rule Based(Custom)') {
          if (values.customFunctionPath && values.customFunctionName) {
            apiConfig.reward = {
              data_source: 'custom',
              custom_reward_function: {
                path: values.customFunctionPath,
                name: values.customFunctionName,
              },
            }
          }
        } else if (rewardMode === 'Reward Model' && values.rewardModelPath) {
          apiConfig.reward = {
            reward_model_path: values.rewardModelPath,
          }
        }

        if (config.trainer.save_freq) apiConfig.trainer.save_freq = config.trainer.save_freq
        if (config.trainer.test_freq !== undefined)
          apiConfig.trainer.test_freq = config.trainer.test_freq
        if (config.trainer.nnodes) apiConfig.trainer.nnodes = config.trainer.nnodes
        if (config.trainer.val_before_train !== undefined)
          apiConfig.trainer.val_before_train = config.trainer.val_before_train
        if (config.trainer.critic_warmup !== undefined)
          apiConfig.trainer.critic_warmup = config.trainer.critic_warmup
      }

      // Handle wandb configuration
      if (hasWandbToken) {
        apiConfig.wandb = {
          api_key: wandbToken,
          entity: null,
          ...(config.method === 'grpo' && { enable_weave: false }),
        }
      }

      const formData = new FormData()
      formData.append('config', JSON.stringify(apiConfig))

      const trainFile = values.trainingData?.[0]?.originFileObj ?? values.trainingData?.[0]
      const valFile =
        values.validationExamples?.[0]?.originFileObj ?? values.validationExamples?.[0]

      if (trainFile) {
        formData.append('train_file', trainFile as Blob)
      }
      if (valFile) {
        formData.append('val_file', valFile as Blob)
      }

      const res = await fetch('/api/train', {
        method: 'POST',
        body: formData,
      })

      if (!res.ok) {
        const text = await res.text()
        throw new Error(text || 'Failed to create training task')
      }

      const data = await res.json()
      if (!data?.job_id) {
        throw new Error('No job_id returned, cannot start training')
      }

      // Save job_id to localStorage for persistence across page refreshes
      const jobState = {
        jobId: data.job_id,
        createdAt: Date.now(),
        taskName: trainingConfig.trainer.experiment_name || 'Training Task',
      }
      localStorage.setItem('training_job_state', JSON.stringify(jobState))

      message.success('Training task created successfully, connecting to log stream')
      onStartTraining?.(data.job_id)
    } catch (error) {
      const err = error as Error
      message.error(err.message || 'Failed to create training task')
    }
    setSubmitting(false)
  }

  const renderSectionContainer = (title: string, children: React.ReactNode) => (
    <div
      style={{
        padding: token.padding,
        background: token.colorFillAlter,
        borderRadius: token.borderRadiusLG,
        border: `1px solid ${token.colorBorderSecondary}`,
      }}
    >
      <div
        style={{
          fontSize: token.fontSize,
          fontWeight: 600,
        }}
      >
        {title}
      </div>
      {children}
    </div>
  )

  return (
    <ConfigProvider
      theme={{
        components: {
          Form: {
            itemMarginBottom: 2,
          },
        },
      }}
    >
      <div className="flex h-full gap-6 px-8">
        {/* Left Column - Basic Config */}
        <div style={{ width: 480, flexShrink: 0 }}>
          <Card
            title={
              <div className="flex items-center gap-2">
                <span
                  className="flex h-6 w-6 items-center justify-center rounded-full"
                  style={{
                    border: `2px solid ${token.colorPrimaryBorder}`,
                    color: token.colorPrimary,
                    fontSize: token.fontSizeSM,
                    fontWeight: 600,
                  }}
                >
                  1
                </span>
                <span>Basic Config</span>
              </div>
            }
            styles={{
              header: {
                borderBottom: 0,
              },
              body: {
                paddingTop: 0,
              },
            }}
          >
            <Form form={form} layout="vertical" size="large" onValuesChange={handleValuesChange}>
              <Form.Item
                name="taskName"
                label="Task Name"
                rules={[{ required: true, message: 'Please enter Task Name' }]}
              >
                <Input placeholder="Enter task name (e.g., task-v1)" size="large" />
              </Form.Item>

              <Form.Item
                name="baseModelPath"
                label="Base Model Path"
                rules={[{ required: true, message: 'Please enter Base Model Path' }]}
              >
                <Input placeholder="/path/to/model" size="large" />
              </Form.Item>

              <Form.Item
                name="wandbToken"
                label={
                  <span>
                    Wandb Token{' '}
                    <span style={{ color: token.colorTextTertiary, fontWeight: 'normal' }}>
                      (Optional)
                    </span>
                  </span>
                }
              >
                <Input placeholder="Enter your Wandb API Key" size="large" />
              </Form.Item>

              <Form.Item
                label={
                  <div
                    style={{
                      display: 'inline-flex',
                      flexDirection: 'column',
                      alignItems: 'flex-start',
                      gap: token.marginXXS,
                    }}
                  >
                    <span
                      style={{
                        display: 'inline-flex',
                        alignItems: 'center',
                        gap: token.marginXXS,
                        lineHeight: '20px',
                      }}
                    >
                      <span style={{ color: token.colorError }}>*</span>
                      <span>Training Data</span>
                    </span>
                    <span style={{ fontSize: token.fontSizeSM, color: token.colorTextTertiary }}>
                      JSONL files for model fine-tuning.
                    </span>
                  </div>
                }
                required={false}
                name="trainingData"
                getValueFromEvent={e => {
                  if (Array.isArray(e)) {
                    return e
                  }
                  return e?.fileList
                }}
                rules={[
                  {
                    validator: (_, value) => {
                      if (!value || !Array.isArray(value) || value.length === 0) {
                        return Promise.reject(new Error('Please upload Training Data'))
                      }
                      return Promise.resolve()
                    },
                  },
                ]}
              >
                <CustomUpload multiple columns={2} accept=".jsonl" />
              </Form.Item>

              <Form.Item
                label={
                  <div
                    style={{
                      display: 'inline-flex',
                      flexDirection: 'column',
                      alignItems: 'flex-start',
                      gap: token.marginXXS,
                    }}
                  >
                    <span
                      style={{
                        display: 'inline-flex',
                        alignItems: 'center',
                        gap: token.marginXXS,
                        lineHeight: '20px',
                      }}
                    >
                      <span style={{ color: token.colorError }}>*</span>
                      <span>Validation Examples</span>
                    </span>
                    <span style={{ fontSize: token.fontSizeSM, color: token.colorTextTertiary }}>
                      JSONL files for validation or demonstration.
                    </span>
                  </div>
                }
                required={false}
                name="validationExamples"
                getValueFromEvent={e => {
                  if (Array.isArray(e)) {
                    return e
                  }
                  return e?.fileList
                }}
                rules={[
                  {
                    validator: (_, value) => {
                      if (!value || !Array.isArray(value) || value.length === 0) {
                        return Promise.reject(new Error('Please upload Validation Examples'))
                      }
                      return Promise.resolve()
                    },
                  },
                ]}
              >
                <CustomUpload multiple columns={2} accept=".jsonl" />
              </Form.Item>
            </Form>
          </Card>
        </div>

        {/* Right Column - Hyperparameters */}
        <div className="flex-1 overflow-y-auto">
          <div className="flex flex-col gap-6">
            <Card
              title={
                <div className="flex items-center gap-2">
                  <span
                    className="flex h-6 w-6 items-center justify-center rounded-full"
                    style={{
                      border: `2px solid ${token.colorPrimaryBorder}`,
                      color: token.colorPrimary,
                      fontSize: token.fontSizeSM,
                      fontWeight: 600,
                    }}
                  >
                    2
                  </span>
                  <span>Hyperparameters</span>
                </div>
              }
              styles={{
                header: {
                  borderBottom: 0,
                },
              }}
            >
              <Form form={form} layout="vertical" size="large" onValuesChange={handleValuesChange}>
                {/* Training Method Selection */}
                <div className="mb-6 flex gap-3">
                  <Button
                    color={trainingMethod === 'sft' ? 'primary' : 'default'}
                    variant="outlined"
                    icon={<SvgIcon icon={SftIcon} size={16} />}
                    onClick={() => setTrainingMethod('sft')}
                    onMouseEnter={() => setHoveredMethod('sft')}
                    onMouseLeave={() => setHoveredMethod(null)}
                    style={{
                      flex: 1,
                      backgroundColor:
                        trainingMethod === 'sft' ? token.colorPrimaryBg : 'transparent',
                      color:
                        trainingMethod === 'sft' || hoveredMethod === 'sft'
                          ? token.colorPrimary
                          : token.colorTextSecondary,
                    }}
                  >
                    SFT
                  </Button>
                  <Button
                    color={trainingMethod === 'grpo' ? 'primary' : 'default'}
                    variant="outlined"
                    icon={<SvgIcon icon={GrpoIcon} size={16} />}
                    onClick={() => setTrainingMethod('grpo')}
                    onMouseEnter={() => setHoveredMethod('grpo')}
                    onMouseLeave={() => setHoveredMethod(null)}
                    style={{
                      flex: 1,
                      backgroundColor:
                        trainingMethod === 'grpo' ? token.colorPrimaryBg : 'transparent',
                      color:
                        trainingMethod === 'grpo' || hoveredMethod === 'grpo'
                          ? token.colorPrimary
                          : token.colorTextSecondary,
                    }}
                  >
                    GRPO
                  </Button>
                </div>

                {/* SFT Fields */}
                {trainingMethod === 'sft' && (
                  <div className="flex flex-col gap-4 px-4">
                    {/* Row 1: Prompt/Response Keys */}
                    <div className="grid grid-cols-2 gap-4">
                      <Form.Item
                        name="promptKey"
                        label="Prompt Key"
                        rules={[{ required: true, message: 'Please enter Prompt Key' }]}
                      >
                        <Input placeholder="e.g., instruction" size="large" />
                      </Form.Item>
                      <Form.Item
                        name="responseKey"
                        label="Response Key"
                        rules={[{ required: true, message: 'Please enter Response Key' }]}
                      >
                        <Input placeholder="e.g., output" size="large" />
                      </Form.Item>
                    </div>

                    {/* Row 2: Learning Rate & Total Epochs */}
                    <div className="grid grid-cols-2 gap-4">
                      <Form.Item
                        name="learningRate"
                        label="Learning Rate"
                        rules={[{ required: true, message: 'Please enter Learning Rate' }]}
                      >
                        <Input placeholder="e.g., 2e-5" size="large" />
                      </Form.Item>
                      <Form.Item
                        name="totalEpochs"
                        label="Total Epochs"
                        rules={[{ required: true, message: 'Please enter Total Epochs' }]}
                      >
                        <NumberStepper min={1} />
                      </Form.Item>
                    </div>

                    {/* Row 3: Train Batch Size & GPUs Per Node */}
                    <div className="grid grid-cols-2 gap-4">
                      <Form.Item
                        name="trainBatchSize"
                        label="Train Batch Size"
                        rules={[{ required: true, message: 'Please enter Train Batch Size' }]}
                      >
                        <Input placeholder="8" size="large" />
                      </Form.Item>
                      <Form.Item
                        name="gpusPerNode"
                        label="GPUs Per Node"
                        rules={[{ required: true, message: 'Please enter GPUs Per Node' }]}
                      >
                        <NumberStepper min={1} />
                      </Form.Item>
                    </div>

                    {/* Row 4: LoRA */}
                    <div className="flex gap-4">
                      <Form.Item
                        name="useLoRAFineTuning"
                        label="Use LoRA Fine-Tuning"
                        valuePropName="checked"
                        rules={[{ required: true }]}
                      >
                        <Switch className="rectangular-switch" />
                      </Form.Item>

                      <Form.Item
                        noStyle
                        shouldUpdate={(prev, curr) =>
                          prev.useLoRAFineTuning !== curr.useLoRAFineTuning
                        }
                      >
                        {({ getFieldValue }) => {
                          const useLoRA = getFieldValue('useLoRAFineTuning')
                          return useLoRA ? (
                            <Form.Item
                              name="loraRank"
                              label="LoRA Rank"
                              rules={[{ required: true, message: 'Please enter LoRA Rank' }]}
                              className="flex-1"
                            >
                              <Input placeholder="8" size="large" />
                            </Form.Item>
                          ) : null
                        }}
                      </Form.Item>
                    </div>
                  </div>
                )}

                {/* GRPO Fields */}
                {trainingMethod === 'grpo' && (
                  <div className="flex flex-col gap-4">
                    {/* Reward Section */}
                    {renderSectionContainer(
                      'Reward',
                      <div
                        className={`grid gap-4 ${
                          rewardMode === 'Rule Based(Custom)' ? 'grid-cols-3' : 'grid-cols-2'
                        }`}
                      >
                        <Form.Item
                          name="rewardMode"
                          label={renderLabelWithTip(
                            'Reward Mode',
                            'Select reward source mode',
                            true,
                          )}
                          rules={[{ required: true, message: 'Please select Reward Mode' }]}
                        >
                          <Select
                            size="large"
                            options={[
                              { label: 'Rule Based(Built in)', value: 'Rule Based(Built in)' },
                              { label: 'Rule Based(Custom)', value: 'Rule Based(Custom)' },
                              { label: 'Reward Model', value: 'Reward Model' },
                            ]}
                          />
                        </Form.Item>

                        <Form.Item
                          noStyle
                          shouldUpdate={(prev, current) => prev.rewardMode !== current.rewardMode}
                        >
                          {({ getFieldValue }) => {
                            const currentMode = getFieldValue('rewardMode')
                            if (currentMode === 'Rule Based(Built in)') {
                              return (
                                <Form.Item
                                  name="dataSource"
                                  label={renderLabelWithTip(
                                    'Data Source',
                                    'Data source for built-in rules',
                                    true,
                                  )}
                                  rules={[{ required: true, message: 'Please enter Data Source' }]}
                                >
                                  <Input placeholder="openai/gsm8k" size="large" />
                                </Form.Item>
                              )
                            }
                            if (currentMode === 'Rule Based(Custom)') {
                              return (
                                <>
                                  <Form.Item
                                    name="customFunctionPath"
                                    label={renderLabelWithTip(
                                      'Function Path',
                                      'Path to custom reward function script',
                                      true,
                                    )}
                                    rules={[
                                      {
                                        required: true,
                                        message: 'Please enter Custom Function Path',
                                      },
                                    ]}
                                  >
                                    <Input placeholder="/path/to/script.py" size="large" />
                                  </Form.Item>
                                  <Form.Item
                                    name="customFunctionName"
                                    label={renderLabelWithTip(
                                      'Function Name',
                                      'Function name in the script',
                                      true,
                                    )}
                                    rules={[
                                      {
                                        required: true,
                                        message: 'Please enter Custom Function Name',
                                      },
                                    ]}
                                  >
                                    <Input placeholder="e.g., compute_reward" size="large" />
                                  </Form.Item>
                                </>
                              )
                            }
                            if (currentMode === 'Reward Model') {
                              return (
                                <Form.Item
                                  name="rewardModelPath"
                                  label={renderLabelWithTip(
                                    'Reward Model Path',
                                    'Local or remote path to the reward model',
                                    true,
                                  )}
                                  rules={[
                                    { required: true, message: 'Please enter Reward Model Path' },
                                  ]}
                                >
                                  <Input placeholder="/path/to/reward_model" size="large" />
                                </Form.Item>
                              )
                            }
                            return null
                          }}
                        </Form.Item>
                      </div>,
                    )}

                    {/* Training Section */}
                    {renderSectionContainer(
                      'Training',
                      <div className="grid grid-cols-4 gap-4">
                        <Form.Item
                          name="learningRate"
                          label="Learning Rate"
                          rules={[{ required: true, message: 'Please enter Learning Rate' }]}
                        >
                          <Input placeholder="e.g., 2e-5" size="large" />
                        </Form.Item>

                        <Form.Item
                          name="totalEpochs"
                          label="Total Epochs"
                          rules={[{ required: true, message: 'Please enter Total Epochs' }]}
                        >
                          <NumberStepper min={1} />
                        </Form.Item>

                        <Form.Item
                          name="numNodes"
                          label="Number of Nodes"
                          rules={[{ required: true, message: 'Please enter Number of Nodes' }]}
                        >
                          <NumberStepper min={1} />
                        </Form.Item>

                        <Form.Item
                          name="gpusPerNode"
                          label="GPUs Per Node"
                          rules={[{ required: true, message: 'Please enter GPUs Per Node' }]}
                        >
                          <NumberStepper min={1} />
                        </Form.Item>
                      </div>,
                    )}

                    {/* Data Section */}
                    {renderSectionContainer(
                      'Data',
                      <div className="flex flex-col gap-x-4">
                        <Form.Item
                          name="trainBatchSize"
                          label="Train Batch Size"
                          rules={[{ required: true, message: 'Please enter Train Batch Size' }]}
                        >
                          <Input placeholder="8" size="large" />
                        </Form.Item>

                        <div className="grid grid-cols-2 gap-4">
                          <Form.Item
                            name="promptKey"
                            label="Prompt Key"
                            rules={[{ required: true, message: 'Please enter Prompt Key' }]}
                          >
                            <Input placeholder="e.g., instruction" size="large" />
                          </Form.Item>
                          <Form.Item
                            name="responseKey"
                            label="Response Key"
                            rules={[{ required: true, message: 'Please enter Response Key' }]}
                          >
                            <Input placeholder="e.g., output" size="large" />
                          </Form.Item>
                        </div>
                      </div>,
                    )}

                    {/* LoRA Section */}
                    {renderSectionContainer(
                      'LoRA',
                      <div className="flex gap-4">
                        <Form.Item
                          name="useLoRAFineTuning"
                          label="Use LoRA Fine-Tuning"
                          valuePropName="checked"
                          rules={[{ required: true }]}
                        >
                          <Switch className="rectangular-switch" />
                        </Form.Item>

                        <Form.Item
                          noStyle
                          shouldUpdate={(prev, curr) =>
                            prev.useLoRAFineTuning !== curr.useLoRAFineTuning
                          }
                        >
                          {({ getFieldValue }) => {
                            const useLoRA = getFieldValue('useLoRAFineTuning')
                            return useLoRA ? (
                              <Form.Item
                                name="loraRank"
                                label="LoRA Rank"
                                rules={[{ required: true, message: 'Please enter LoRA Rank' }]}
                                className="flex-1"
                              >
                                <Input placeholder="8" size="large" />
                              </Form.Item>
                            ) : null
                          }}
                        </Form.Item>
                      </div>,
                    )}
                  </div>
                )}
                <Divider />

                {/* Advanced Section */}
                <Collapse
                  ghost
                  styles={{
                    header: {
                      padding: 0,
                    },
                  }}
                  expandIconPosition="end"
                  items={[
                    {
                      key: '1',
                      label: (
                        <span
                          style={{
                            fontSize: token.fontSizeLG,
                            fontWeight: 500,
                            color: token.colorPrimary,
                          }}
                        >
                          Advanced
                        </span>
                      ),
                      children: (
                        <>
                          {trainingMethod === 'sft' && (
                            <div className="mb-4 grid grid-cols-2 gap-x-4">
                              <Form.Item
                                name="truncation"
                                label={
                                  <span>
                                    Truncation{' '}
                                    <span
                                      style={{
                                        color: token.colorTextTertiary,
                                        fontWeight: 'normal',
                                      }}
                                    >
                                      (Optional)
                                    </span>
                                  </span>
                                }
                              >
                                <Input placeholder="right" size="large" />
                              </Form.Item>

                              <Form.Item
                                name="maxLength"
                                label={
                                  <span>
                                    Max Length{' '}
                                    <span
                                      style={{
                                        color: token.colorTextTertiary,
                                        fontWeight: 'normal',
                                      }}
                                    >
                                      (Optional)
                                    </span>
                                  </span>
                                }
                              >
                                <Input placeholder="2048" size="large" />
                              </Form.Item>

                              <Form.Item
                                name="microBatchSize"
                                label={
                                  <span>
                                    Micro Batch Size{' '}
                                    <span
                                      style={{
                                        color: token.colorTextTertiary,
                                        fontWeight: 'normal',
                                      }}
                                    >
                                      (Optional)
                                    </span>
                                  </span>
                                }
                              >
                                <Input placeholder="1" size="large" />
                              </Form.Item>

                              <Form.Item
                                name="numNodes"
                                label={
                                  <span>
                                    Number of Nodes{' '}
                                    <span
                                      style={{
                                        color: token.colorTextTertiary,
                                        fontWeight: 'normal',
                                      }}
                                    >
                                      (Optional)
                                    </span>
                                  </span>
                                }
                              >
                                <NumberStepper min={1} />
                              </Form.Item>
                            </div>
                          )}

                          {trainingMethod === 'grpo' && (
                            <div className="mb-4 grid grid-cols-2 gap-x-4">
                              <Form.Item
                                name="rolloutGpuMemoryUtil"
                                label={
                                  <span>
                                    Rollout GPU Memory Util{' '}
                                    <span
                                      style={{
                                        color: token.colorTextTertiary,
                                        fontWeight: 'normal',
                                      }}
                                    >
                                      (Optional)
                                    </span>
                                  </span>
                                }
                              >
                                <Input placeholder="0.9" size="large" />
                              </Form.Item>

                              <Form.Item
                                name="rolloutN"
                                label={
                                  <span>
                                    Rollout N{' '}
                                    <span
                                      style={{
                                        color: token.colorTextTertiary,
                                        fontWeight: 'normal',
                                      }}
                                    >
                                      (Optional)
                                    </span>
                                  </span>
                                }
                              >
                                <NumberStepper min={1} />
                              </Form.Item>

                              <Form.Item
                                name="maxPromptLength"
                                label={
                                  <span>
                                    Max Prompt Length{' '}
                                    <span
                                      style={{
                                        color: token.colorTextTertiary,
                                        fontWeight: 'normal',
                                      }}
                                    >
                                      (Optional)
                                    </span>
                                  </span>
                                }
                              >
                                <Input placeholder="1024" size="large" />
                              </Form.Item>

                              <Form.Item
                                name="maxResponseLength"
                                label={
                                  <span>
                                    Max Response Length{' '}
                                    <span
                                      style={{
                                        color: token.colorTextTertiary,
                                        fontWeight: 'normal',
                                      }}
                                    >
                                      (Optional)
                                    </span>
                                  </span>
                                }
                              >
                                <Input placeholder="1024" size="large" />
                              </Form.Item>

                              <Form.Item
                                name="saveFrequency"
                                label={
                                  <span>
                                    Save Frequency (steps){' '}
                                    <span
                                      style={{
                                        color: token.colorTextTertiary,
                                        fontWeight: 'normal',
                                      }}
                                    >
                                      (Optional)
                                    </span>
                                  </span>
                                }
                              >
                                <InputNumber
                                  placeholder="100"
                                  style={{ width: '100%' }}
                                  size="large"
                                />
                              </Form.Item>

                              <Form.Item
                                name="testFrequency"
                                label={
                                  <span>
                                    Test Frequency (steps){' '}
                                    <span
                                      style={{
                                        color: token.colorTextTertiary,
                                        fontWeight: 'normal',
                                      }}
                                    >
                                      (Optional)
                                    </span>
                                  </span>
                                }
                              >
                                <InputNumber
                                  placeholder="100"
                                  style={{ width: '100%' }}
                                  size="large"
                                />
                              </Form.Item>

                              <Form.Item
                                name="ppoMiniBatchSize"
                                label={
                                  <span>
                                    PPO Mini Batch Size{' '}
                                    <span
                                      style={{
                                        color: token.colorTextTertiary,
                                        fontWeight: 'normal',
                                      }}
                                    >
                                      (Optional)
                                    </span>
                                  </span>
                                }
                                dependencies={['trainBatchSize']}
                                rules={[
                                  ({ getFieldValue }) => ({
                                    validator(_, value) {
                                      const trainBatchSize = getFieldValue('trainBatchSize')
                                      if (!value || !trainBatchSize) {
                                        return Promise.resolve()
                                      }
                                      if (Number(value) >= Number(trainBatchSize)) {
                                        return Promise.reject(
                                          new Error(
                                            'PPO Mini Batch Size must be less than Train Batch Size',
                                          ),
                                        )
                                      }
                                      return Promise.resolve()
                                    },
                                  }),
                                ]}
                              >
                                <Input placeholder="1" size="large" />
                              </Form.Item>

                              <Form.Item
                                name="ppoMicroBatchSize"
                                label={
                                  <span>
                                    PPO Micro Batch Size{' '}
                                    <span
                                      style={{
                                        color: token.colorTextTertiary,
                                        fontWeight: 'normal',
                                      }}
                                    >
                                      (Optional)
                                    </span>
                                  </span>
                                }
                              >
                                <Input placeholder="1" size="large" />
                              </Form.Item>

                              <Form.Item
                                name="useKLLoss"
                                label={
                                  <span>
                                    Use KL Loss{' '}
                                    <span
                                      style={{
                                        color: token.colorTextTertiary,
                                        fontWeight: 'normal',
                                      }}
                                    >
                                      (Optional)
                                    </span>
                                  </span>
                                }
                                valuePropName="checked"
                              >
                                <Switch className="rectangular-switch" />
                              </Form.Item>

                              <Form.Item
                                noStyle
                                shouldUpdate={(prev, current) =>
                                  prev.useKLLoss !== current.useKLLoss
                                }
                              >
                                {({ getFieldValue }) =>
                                  getFieldValue('useKLLoss') === true ? (
                                    <Form.Item
                                      name="klLossCoefficient"
                                      label={
                                        <span>
                                          KL Loss Coefficient{' '}
                                          <span
                                            style={{
                                              color: token.colorTextTertiary,
                                              fontWeight: 'normal',
                                            }}
                                          >
                                            (Optional)
                                          </span>
                                        </span>
                                      }
                                    >
                                      <InputNumber
                                        placeholder="0.1"
                                        style={{ width: '100%' }}
                                        size="large"
                                      />
                                    </Form.Item>
                                  ) : null
                                }
                              </Form.Item>
                            </div>
                          )}
                        </>
                      ),
                    },
                  ]}
                />

                <Divider />

                {/* Save & Continue Button */}
                <div
                  style={{
                    display: 'flex',
                    justifyContent: 'flex-end',
                  }}
                >
                  <Button
                    type="primary"
                    size="large"
                    onClick={handleSaveAndContinue}
                    loading={submitting}
                  >
                    Save & Continue
                  </Button>
                </div>
              </Form>
            </Card>
          </div>
        </div>
      </div>
    </ConfigProvider>
  )
}
