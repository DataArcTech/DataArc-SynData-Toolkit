import { EventStreamContentType, fetchEventSource } from '@microsoft/fetch-event-source'
import { Button, Card, Collapse, Form, Input, message, Select } from 'antd'
import { useCallback, useEffect, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { taskApi } from '@/api/task'
import DistillIcon from '@/assets/icon/distill.svg?react'
import LocalIcon from '@/assets/icon/local.svg?react'
import WebIcon from '@/assets/icon/web.svg?react'
import { CustomUpload } from '@/components/custom-upload'
import GenerationCoreModal from '@/components/generation-core-modal'
import SvgIcon from '@/components/svg-icon'
import { useAntdTheme } from '@/hooks/use-antd-theme'
import { type SSEEventData, useTaskStore } from '@/store/use-task-store'

const { TextArea } = Input

type TaskType = 'local' | 'web' | 'distill'

export default function ConfigurationPage() {
  const token = useAntdTheme()
  const navigate = useNavigate()
  const [form] = Form.useForm()
  const [taskType, setTaskType] = useState<TaskType>('local')
  const [hoveredTaskType, setHoveredTaskType] = useState<TaskType | null>(null)

  const [modalOpen, setModalOpen] = useState(false)
  const [modalInitialTab, setModalInitialTab] = useState<
    'llm-configuration' | 'evaluation-metrics' | 'execution-environment'
  >('llm-configuration')

  const {
    config,
    setIsGenerating,
    isGenerating,
    setEventData,
    setTaskConfig,
    hasAutoOpenedConfigModal,
    setHasAutoOpenedConfigModal,
  } = useTaskStore()
  const [isConnected, setIsConnected] = useState(false)
  const abortControllerRef = useRef<AbortController | null>(null)
  const shouldStopRef = useRef(false)
  const connectionCountRef = useRef(0)

  const isSettingFormValuesRef = useRef(false)

  const language = Form.useWatch('language', form) || 'english'

  // biome-ignore lint/correctness/useExhaustiveDependencies: Only initialize on mount
  useEffect(() => {
    if (config.task.task_type) {
      setTaskType(config.task.task_type as TaskType)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // biome-ignore lint/correctness/useExhaustiveDependencies: Only initialize form on mount, not on config changes
  useEffect(() => {
    const initialValues = {
      taskName: config.task.name || '',
      taskInstruction: config.task.task_instruction || '',
      numberOfExamples: config.task.num_samples || 10,
      language: config.translation.language || 'english',
      answerExtractionTag: config.answer_extraction.tag || '',
      answerExtractionInstruction: config.answer_extraction.instruction || '',
      domain: config.task.domain || '',
      inputInstruction: config.task.input_instruction || '',
      outputInstruction: config.task.output_instruction || '',
      parserMethod: config.task.local.parsing.method || '',
      huggingfaceToken: config.task.web.huggingface_token || '',
      datasetScoreThreshold: config.task.web.dataset_score_threshold || 30,
      arabicTranslatorModelPath: config.translation.model_path || '',
    }

    isSettingFormValuesRef.current = true
    form.setFieldsValue(initialValues)
    setTimeout(() => {
      isSettingFormValuesRef.current = false
    }, 0)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // biome-ignore lint/correctness/useExhaustiveDependencies: Only check on mount
  useEffect(() => {
    if (!hasAutoOpenedConfigModal) {
      setModalOpen(true)
      setHasAutoOpenedConfigModal(true)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const handleValuesChange = (_changedValues: unknown, allValues: Record<string, unknown>) => {
    if (isSettingFormValuesRef.current) {
      return
    }

    const updates: Partial<typeof config> = {}

    if (allValues.taskName !== undefined) {
      updates.task = {
        ...config.task,
        name: allValues.taskName as string,
      }
    }

    if (allValues.taskInstruction !== undefined) {
      updates.task = {
        ...(updates.task || config.task),
        task_instruction: allValues.taskInstruction as string,
      }
    }

    if (allValues.numberOfExamples !== undefined) {
      updates.task = {
        ...(updates.task || config.task),
        num_samples: Number(allValues.numberOfExamples) || 0,
      }
    }

    if (allValues.language !== undefined) {
      updates.translation = {
        ...config.translation,
        language: allValues.language as string,
      }
    }

    if (allValues.arabicTranslatorModelPath !== undefined) {
      updates.translation = {
        ...(updates.translation || config.translation),
        model_path: allValues.arabicTranslatorModelPath as string,
      }
    }

    if (allValues.answerExtractionTag !== undefined) {
      updates.answer_extraction = {
        ...config.answer_extraction,
        tag: allValues.answerExtractionTag as string,
      }
    }

    if (allValues.answerExtractionInstruction !== undefined) {
      updates.answer_extraction = {
        ...(updates.answer_extraction || config.answer_extraction),
        instruction: allValues.answerExtractionInstruction as string,
      }
    }

    if (allValues.domain !== undefined) {
      updates.task = {
        ...(updates.task || config.task),
        domain: allValues.domain as string,
      }
    }

    if (allValues.inputInstruction !== undefined) {
      updates.task = {
        ...(updates.task || config.task),
        input_instruction: allValues.inputInstruction as string,
      }
    }

    if (allValues.outputInstruction !== undefined) {
      updates.task = {
        ...(updates.task || config.task),
        output_instruction: allValues.outputInstruction as string,
      }
    }

    if (allValues.parserMethod !== undefined) {
      updates.task = {
        ...(updates.task || config.task),
        local: {
          ...config.task.local,
          parsing: {
            ...config.task.local.parsing,
            method: allValues.parserMethod as string,
          },
        },
      }
    }

    if (allValues.huggingfaceToken !== undefined) {
      updates.task = {
        ...(updates.task || config.task),
        web: {
          ...config.task.web,
          huggingface_token: allValues.huggingfaceToken as string,
        },
      }
    }

    if (allValues.datasetScoreThreshold !== undefined) {
      updates.task = {
        ...(updates.task || config.task),
        web: {
          ...(updates.task?.web || config.task.web),
          dataset_score_threshold: Number(allValues.datasetScoreThreshold) || 30,
        },
      }
    }

    if (Object.keys(updates).length > 0) {
      setTaskConfig(updates)
    }
  }

  // biome-ignore lint/correctness/useExhaustiveDependencies: Only update on taskType change
  useEffect(() => {
    setTaskConfig({
      task: {
        ...config.task,
        task_type: taskType,
      },
    })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [taskType])

  // Set initial values
  const initialValues = {
    language: 'english',
    parserMethod: 'mineru',
    datasetScoreThreshold: 30,
  }

  const taskTypeButtons = [
    {
      key: 'local' as TaskType,
      label: 'Local',
      icon: LocalIcon,
    },
    {
      key: 'web' as TaskType,
      label: 'Web',
      icon: WebIcon,
    },
    {
      key: 'distill' as TaskType,
      label: 'Distill',
      icon: DistillIcon,
    },
  ]

  // 验证 Generation Core Modal 配置
  const validateGenerationCoreConfig = (): {
    isValid: boolean
    invalidTab?: 'llm-configuration' | 'evaluation-metrics' | 'execution-environment'
    message?: string
  } => {
    // 1. 验证 LLM Configuration
    if (!config.llm.provider || !config.llm.model || !config.llm.api_key) {
      return {
        isValid: false,
        invalidTab: 'llm-configuration',
        message: 'Please complete LLM Configuration (Provider, Model, API Key are required)',
      }
    }

    // 2. 验证 Evaluation Metrics
    if (!config.postprocess.majority_voting.method || !config.evaluation.answer_comparison.method) {
      return {
        isValid: false,
        invalidTab: 'evaluation-metrics',
        message:
          'Please complete Evaluation Metrics (Majority Voting Method and Answer Comparison Method are required)',
      }
    }

    // 检查 Base Model Path
    if (!config.base_model.path) {
      return {
        isValid: false,
        invalidTab: 'evaluation-metrics',
        message: 'Base Model Path is required',
      }
    }

    // 检查 Batch Size
    if (!config.task.batch_size || config.task.batch_size < 1) {
      return {
        isValid: false,
        invalidTab: 'evaluation-metrics',
        message: 'Batch Size is required and must be greater than 0',
      }
    }

    // 检查是否需要 Semantic Model Path（当任意一个为 semantic 时）
    if (
      (config.postprocess.majority_voting.method === 'semantic' ||
        config.evaluation.answer_comparison.method === 'semantic') &&
      !config.evaluation.answer_comparison.semantic?.model_path
    ) {
      return {
        isValid: false,
        invalidTab: 'evaluation-metrics',
        message: 'Semantic Model Path is required when using semantic method',
      }
    }

    // 3. 验证 Execution Environment
    if (!config.device || !config.output_dir) {
      return {
        isValid: false,
        invalidTab: 'execution-environment',
        message:
          'Please complete Execution Environment (CUDA Device and Output Directory are required)',
      }
    }

    return { isValid: true }
  }

  // SSE 连接函数（未使用，但保留以备将来使用）
  // @ts-expect-error - Function is intentionally unused but kept for future use
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const _connectSSE = async (jobId: string) => {
    const controller = new AbortController()
    abortControllerRef.current = controller
    setIsConnected(true)

    await fetchEventSource(`/api/sdg/${jobId}/generate`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      signal: controller.signal,
      openWhenHidden: false,

      async onopen(response) {
        if (shouldStopRef.current) {
          throw new Error('Connection stopped by user')
        }

        if (response.ok && response.headers.get('content-type') === EventStreamContentType) {
          return
        }

        throw new Error(`Failed to connect: ${response.statusText}`)
      },

      onmessage(event) {
        try {
          const data = JSON.parse(event.data) as SSEEventData

          setEventData(data)

          if (data.status === 'generation_complete' || data.status === 'completed') {
            message.success('Data generation completed')
            setIsGenerating(false)
            controller.abort()
            // Navigate to generate page
            navigate('/generate')
          }
        } catch (_error) {
          // Error parsing SSE data
        }
      },

      onerror(err) {
        if (shouldStopRef.current) {
          throw new Error('Connection stopped by user')
        }

        connectionCountRef.current++

        if (connectionCountRef.current > 3) {
          message.error('Too many connection failures, stopped reconnecting')
          setIsConnected(false)
          throw new Error('Max retries exceeded')
        }

        throw err
      },

      onclose() {
        setIsConnected(false)
      },
    })
  }

  // 停止 SSE 连接
  const stopSSE = useCallback(() => {
    shouldStopRef.current = true
    if (abortControllerRef.current) {
      abortControllerRef.current.abort()
      abortControllerRef.current = null
    }
    setIsConnected(false)
  }, [])

  // 处理开始生成
  const handleStartGeneration = async () => {
    try {
      // 1. Validate page form first and get form values
      const formValues = await form.validateFields()

      // 2. Validate Generation Core Modal config
      const coreValidation = validateGenerationCoreConfig()
      if (!coreValidation.isValid) {
        message.error(coreValidation.message || 'Please complete Generation Core configuration')
        // Open modal and navigate to the invalid tab
        if (coreValidation.invalidTab) {
          setModalInitialTab(coreValidation.invalidTab)
        }
        setModalOpen(true)
        return
      }

      // 3. Collect all files from form
      const allFiles: File[] = []

      // From uploadDocuments field
      if (formValues.uploadDocuments && Array.isArray(formValues.uploadDocuments)) {
        formValues.uploadDocuments.forEach((uploadFile: any) => {
          if (uploadFile.originFileObj) {
            allFiles.push(uploadFile.originFileObj)
          }
        })
      }

      // From demoExamples field
      if (formValues.demoExamples && Array.isArray(formValues.demoExamples)) {
        formValues.demoExamples.forEach((uploadFile: any) => {
          if (uploadFile.originFileObj) {
            allFiles.push(uploadFile.originFileObj)
          }
        })
      }

      // 4. Reset all previous task state before starting new task
      const { resetTask, resetStats } = useTaskStore.getState()
      resetTask() // Clear eventData, refineEventData, stepHistory, etc.
      resetStats() // Clear timer and token statistics

      // 5. Start generation
      setIsGenerating(true)

      // Create job with config and files
      const response = await taskApi.createJob({
        config,
        files: allFiles.length > 0 ? allFiles : undefined,
      })
      const jobId = response.job_id

      if (!jobId) {
        throw new Error('Failed to get job_id')
      }

      message.success(`Task created: ${jobId}`)

      // Save task ID to store and navigate to generate page
      // The generate page will establish SSE connection
      const { setTaskId } = useTaskStore.getState()
      setTaskId(jobId)

      // Navigate to generate page immediately
      navigate('/generate')
    } catch (err) {
      const error = err instanceof Error ? err : new Error(String(err))
      message.error(`Failed to start task: ${error.message}`)
      setIsGenerating(false)
    }
  }

  // 处理停止生成
  const handleStopGeneration = () => {
    stopSSE()
    setIsGenerating(false)
    message.info('Task stopped')
  }

  const renderSourceDataContent = () => {
    switch (taskType) {
      case 'local':
        return (
          <>
            <Form.Item
              label="Upload Documents"
              name="uploadDocuments"
              extra={
                <div
                  style={{
                    fontSize: token.fontSizeSM,
                    color: token.colorTextTertiary,
                  }}
                >
                  Upload local documents for data generation.
                </div>
              }
              getValueFromEvent={e => {
                if (Array.isArray(e)) {
                  return e
                }
                return e?.fileList
              }}
              rules={[
                {
                  required: true,
                  validator: (_, value) => {
                    if (!value || !Array.isArray(value) || value.length === 0) {
                      return Promise.reject(new Error('Please upload documents'))
                    }
                    return Promise.resolve()
                  },
                },
              ]}
            >
              <CustomUpload multiple columns={2} accept=".pdf" />
            </Form.Item>

            {/* Hidden Parser Method field with default value */}
            <Form.Item name="parserMethod" hidden>
              <Input />
            </Form.Item>

            <Form.Item
              label="Demo Examples (Optional)"
              name="demoExamples"
              extra={
                <div
                  style={{
                    fontSize: token.fontSizeSM,
                    color: token.colorTextTertiary,
                  }}
                >
                  Upload JSONL file(s) to serve as a generation example.
                </div>
              }
              getValueFromEvent={e => {
                if (Array.isArray(e)) {
                  return e
                }
                return e?.fileList
              }}
            >
              <CustomUpload multiple accept=".jsonl" columns={2} />
            </Form.Item>
          </>
        )
      case 'web':
        return (
          <>
            <Form.Item name="huggingfaceToken" label="Huggingface Token (Optional)">
              <div
                style={{
                  marginTop: -token.marginXS,
                }}
              >
                <div
                  style={{
                    fontSize: token.fontSizeSM,
                    color: token.colorTextTertiary,
                    marginBottom: token.marginXS,
                  }}
                >
                  Enter your Huggingface access token.
                </div>
                <Input placeholder="hf_..." size="large" />
              </div>
            </Form.Item>

            <Form.Item
              name="datasetScoreThreshold"
              label="Dataset Score Threshold"
              rules={[{ required: true, message: 'Please enter Dataset Score Threshold' }]}
            >
              <Input placeholder="30" type="number" size="large" />
            </Form.Item>
          </>
        )
      case 'distill':
        return (
          <Form.Item
            label="Demo Examples (Optional)"
            name="demoExamples"
            extra={
              <div
                style={{
                  fontSize: token.fontSizeSM,
                  color: token.colorTextTertiary,
                }}
              >
                Upload JSONL file(s) to serve as a generation example.
              </div>
            }
            getValueFromEvent={e => {
              if (Array.isArray(e)) {
                return e
              }
              return e?.fileList
            }}
          >
            <CustomUpload multiple accept=".jsonl" columns={2} />
          </Form.Item>
        )
      default:
        return null
    }
  }

  return (
    <div className="flex h-[calc(100vh-64px)]" style={{ background: token.colorBgLayout }}>
      <div className="mx-auto flex w-full gap-6 p-8 pb-0" style={{ maxWidth: 1280 }}>
        {/* Left Column - Fixed 480px */}
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
                <span>Source Data Configuration</span>
              </div>
            }
            styles={{
              header: {
                borderBottom: 0,
              },
            }}
          >
            <Form form={form} layout="vertical" size="large" onValuesChange={handleValuesChange}>
              <div className="mb-6 flex gap-3">
                {taskTypeButtons.map(btn => (
                  <Button
                    key={btn.key}
                    color={taskType === btn.key ? 'primary' : 'default'}
                    variant="outlined"
                    icon={<SvgIcon icon={btn.icon} />}
                    onClick={() => setTaskType(btn.key)}
                    onMouseEnter={() => setHoveredTaskType(btn.key)}
                    onMouseLeave={() => setHoveredTaskType(null)}
                    style={{
                      flex: 1,
                      backgroundColor: taskType === btn.key ? token.colorPrimaryBg : 'transparent',
                      color:
                        taskType === btn.key || hoveredTaskType === btn.key
                          ? token.colorPrimary
                          : token.colorTextSecondary,
                    }}
                  >
                    {btn.label}
                  </Button>
                ))}
              </div>

              {renderSourceDataContent()}
            </Form>
          </Card>
        </div>

        {/* Right Column - Scrollable */}
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
                  <span>Task Configuration</span>
                </div>
              }
              styles={{
                header: {
                  borderBottom: 0,
                },
              }}
            >
              <Form
                form={form}
                layout="vertical"
                size="large"
                initialValues={initialValues}
                onValuesChange={handleValuesChange}
              >
                <Form.Item
                  name="taskName"
                  label="Task Name"
                  rules={[{ required: true, message: 'Please enter Task Name' }]}
                >
                  <Input placeholder="custom_task" size="large" />
                </Form.Item>

                <Form.Item
                  name="taskInstruction"
                  label="Task Instruction"
                  rules={[{ required: true, message: 'Please enter Task Instruction' }]}
                >
                  <TextArea placeholder="What kind of data to generate?" rows={4} size="large" />
                </Form.Item>

                {/* Number of Examples and Language in one row */}
                <div className="grid grid-cols-2 gap-4">
                  <Form.Item
                    name="numberOfExamples"
                    label="Number of Samples"
                    rules={[{ required: true, message: 'Please enter Number of Samples' }]}
                  >
                    <Input placeholder="10" type="number" size="large" />
                  </Form.Item>

                  <Form.Item
                    name="language"
                    label="Dataset Language"
                    rules={[{ required: true, message: 'Please select Dataset Language' }]}
                  >
                    <Select
                      placeholder="Select language"
                      size="large"
                      options={[
                        { label: 'English', value: 'english' },
                        { label: 'Arabic', value: 'arabic' },
                      ]}
                    />
                  </Form.Item>
                </div>

                {language === 'arabic' && (
                  <Form.Item
                    name="arabicTranslatorModelPath"
                    label="Arabic Translator Model Path"
                    rules={[
                      { required: true, message: 'Please enter Arabic Translator Model Path' },
                    ]}
                  >
                    <Input placeholder="/path/to/translator/model" size="large" />
                  </Form.Item>
                )}

                {/* Answer extraction in one row */}
                <div className="grid grid-cols-2 gap-4">
                  <Form.Item
                    name="answerExtractionTag"
                    label="Answer extraction (tag)"
                    rules={[{ required: true, message: 'Please enter Answer extraction tag' }]}
                  >
                    <Input placeholder="####" size="large" />
                  </Form.Item>

                  <Form.Item
                    name="answerExtractionInstruction"
                    label="Answer extraction (instruction)"
                    rules={[
                      { required: true, message: 'Please enter Answer extraction instruction' },
                    ]}
                  >
                    <Input placeholder="Output your final answer after ####" size="large" />
                  </Form.Item>
                </div>
              </Form>
            </Card>

            {/* Optional Prompt Card */}
            <Collapse
              ghost
              className="optional-prompt-collapse"
              styles={{
                header: {
                  background: token.colorBgContainer,
                  borderRadius: token.borderRadiusLG,
                  border: `1px solid ${token.colorBorderSecondary}`,
                  padding: `${token.paddingSM}px ${token.padding}px`,
                },
                body: {
                  background: token.colorBgContainer,
                  border: `1px solid ${token.colorBorderSecondary}`,
                  borderTop: 'none',
                  borderRadius: `0 0 ${token.borderRadiusLG}px ${token.borderRadiusLG}px`,
                  padding: token.padding,
                },
              }}
              items={[
                {
                  key: '1',
                  label: (
                    <span style={{ fontSize: token.fontSizeLG, fontWeight: 500 }}>
                      Optional Prompt
                    </span>
                  ),
                  children: (
                    <Form
                      form={form}
                      layout="vertical"
                      size="large"
                      onValuesChange={handleValuesChange}
                    >
                      <Form.Item name="domain" label="Domain">
                        <Input placeholder="Enter domain" size="large" />
                      </Form.Item>

                      <Form.Item name="inputInstruction" label="Input Instruction">
                        <TextArea placeholder="Enter input instruction" rows={4} size="large" />
                      </Form.Item>

                      <Form.Item name="outputInstruction" label="Output Instruction">
                        <TextArea placeholder="Enter output instruction" rows={4} size="large" />
                      </Form.Item>
                    </Form>
                  ),
                },
              ]}
            />

            {/* Start Generate Button */}
            <div
              style={{
                padding: `${token.padding}px`,
                display: 'flex',
                justifyContent: 'flex-end',
                gap: token.margin,
              }}
            >
              {!isGenerating ? (
                <Button type="primary" size="large" onClick={handleStartGeneration}>
                  Start Generate
                </Button>
              ) : (
                <Button danger size="large" onClick={handleStopGeneration}>
                  Stop Generation
                </Button>
              )}
              {isConnected && (
                <span style={{ color: token.colorSuccess, lineHeight: '40px' }}>● Connecting</span>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Generation Core Modal */}
      <GenerationCoreModal
        open={modalOpen}
        onCancel={() => setModalOpen(false)}
        initialTab={modalInitialTab}
      />
    </div>
  )
}
