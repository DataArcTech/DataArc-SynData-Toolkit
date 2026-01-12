import { create } from 'zustand'
import { devtools } from 'zustand/middleware'

// 任务配置接口
export interface TaskConfig {
  device: string
  n_workers: number
  output_dir: string
  export_format: string
  task: TaskInfo
  base_model: BaseModelConfig
  llm: LLMConfig
  answer_extraction: AnswerExtractionConfig
  postprocess: PostprocessConfig
  evaluation: EvaluationConfig
  rewrite: RewriteConfig
  translation: TranslationConfig
}

export interface TaskInfo {
  name: string
  domain: string
  task_instruction: string
  input_instruction: string
  output_instruction: string
  num_samples: number
  batch_size: number
  demo_examples_path: string
  text: TextModalityConfig
}

export interface TextModalityConfig {
  local?: LocalConfig | null
  web?: WebConfig | null
  distill?: DistillConfig | null
}

export interface LocalConfig {
  parsing: ParsingConfig
  retrieval: RetrievalConfig
  generation: GenerationConfig
}

export interface ParsingConfig {
  method: string
}

export interface RetrievalConfig {
  passages_dir: string
  method: string
  top_k: number
}

export interface GenerationConfig {
  temperature: number
}

export interface DistillConfig {
  temperature: number
}

export interface WebConfig {
  huggingface_token: string
  dataset_limit?: number
}

export interface BaseModelConfig {
  provider: string
  path: string
}

export interface LLMConfig {
  provider: string
  model: string
  api_key: string
  base_url: string
}

export interface AnswerExtractionConfig {
  tag: string
  instruction: string
}

export interface PostprocessConfig {
  methods: string[]
  majority_voting: MajorityVotingConfig
}

export interface MajorityVotingConfig {
  n: number
  method: string
  semantic_clustering?: {
    model_path: string
  }
}

export interface EvaluationConfig {
  answer_comparison: AnswerComparisonConfig
}

export interface AnswerComparisonConfig {
  method: string
  semantic?: {
    model_path: string
  }
}

export interface RewriteConfig {
  method: string
}

export interface TranslationConfig {
  language: string
  model_path: string | null
}

// SSE 事件数据类型
export interface SSEPhase {
  id: string
  name: string
  status: 'pending' | 'running' | 'completed' | 'error' | 'failed'
  current_step_index: number
  total_steps: number
}

export interface SSEUsage {
  tokens: number
  time: number
  estimated_remaining_tokens: number
  estimated_remaining_time: number
}

export interface SSEProgress {
  completed: number
  total: number
  unit: string
}

export interface SSEBatch {
  current: number
  total: number
  size: number
}

export interface SSEStep {
  id: string
  name: string
  status: 'running' | 'completed' // 后端步骤状态
  message: string | null
  progress: SSEProgress | null
  batch: SSEBatch | null
  current_item: any
  usage: SSEUsage | null
  result: any
}

export interface SSEEventData {
  job_id: string
  task_type: string
  task_name: string
  status:
  | 'pending'
  | 'running'
  | 'generation_complete'
  | 'complete'
  | 'completed'
  | 'error'
  | 'failed'
  created_at: string
  updated_at: string
  phase: SSEPhase | null
  step: SSEStep | null
  error: {
    code: string
    message: string
    details?: any
  } | null
  output: any
}

// 步骤历史记录（用于展示所有步骤的进度）
export interface StepHistory {
  [compositeStepId: string]: {
    id: string
    phaseId: string
    name: string
    message: string | null
    status: 'pending' | 'loading' | 'success' | 'error'
    usage: SSEUsage | null
    result: any
  }
}

interface TaskState {
  // 任务配置信息
  config: TaskConfig

  // 任务ID，生成开始后由后端返回
  taskId: string | null

  // 任务是否正在生成中
  isGenerating: boolean

  // SSE 事件数据
  eventData: SSEEventData | null

  // 步骤历史记录
  stepHistory: StepHistory

  // 新增：当前所处阶段
  currentPhase: 'generation' | 'refine' | null

  // 新增：Refine 阶段的事件数据
  refineEventData: SSEEventData | null

  // 新增：是否正在执行 Refine
  isRefining: boolean

  // 新增：Generation 是否已完成
  generationCompleted: boolean

  // 累加统计（跨阶段累加，只在创建任务时重置）
  accumulatedTokens: number // 累加的总 token 数量
  timerStartTime: number | null // 计时器开始的时间戳
  pausedDuration: number // 累计暂停时长（毫秒）
  lastPauseTime: number | null // 最后一次暂停的时间戳

  // Configuration 页面状态
  hasAutoOpenedConfigModal: boolean // 是否已自动打开过配置模态框

  // SSE 连接状态
  generationSSEConnected: boolean // Generation SSE 连接状态
  refineSSEConnected: boolean // Refine SSE 连接状态

  // SSE 连接追踪（防止重复连接）
  hasConnectedGeneration: boolean // 是否已建立过 Generation SSE 连接
  hasConnectedRefine: boolean // 是否已建立过 Refine SSE 连接

  // Actions
  setTaskConfig: (config: Partial<TaskConfig>) => void
  setTaskId: (id: string | null) => void
  setIsGenerating: (isGenerating: boolean) => void
  setEventData: (data: SSEEventData) => void
  getTargetProgress: (eventData: SSEEventData | null) => number
  resetTask: () => void

  // 新增 Actions
  setCurrentPhase: (phase: 'generation' | 'refine' | null) => void
  setRefineEventData: (data: SSEEventData) => void
  setIsRefining: (isRefining: boolean) => void
  setGenerationCompleted: (completed: boolean) => void

  // 统计相关 Actions
  startTimer: () => void
  pauseTimer: () => void
  resumeTimer: () => void
  resetStats: () => void
  getCurrentElapsedTime: () => number // 获取当前已累计的时间（秒）

  // Configuration 页面 Actions
  setHasAutoOpenedConfigModal: (hasOpened: boolean) => void

  // SSE 连接状态 Actions
  setGenerationSSEConnected: (connected: boolean) => void
  setRefineSSEConnected: (connected: boolean) => void
  setHasConnectedGeneration: (connected: boolean) => void
  setHasConnectedRefine: (connected: boolean) => void
}

const DEFAULT_CONFIG: TaskConfig = {
  device: 'cuda:0',
  n_workers: 2,
  output_dir: './outputs',
  export_format: 'jsonl',
  task: {
    name: '',
    domain: '',
    task_instruction: '',
    input_instruction: '',
    output_instruction: '',
    num_samples: 10,
    batch_size: 5,
    demo_examples_path: '',
    text: {},
  },
  base_model: {
    provider: 'local',
    path: '',
  },
  llm: {
    provider: 'openai',
    model: '',
    api_key: '',
    base_url: '',
  },
  answer_extraction: {
    tag: '####',
    instruction: 'Output your final answer after ####',
  },
  postprocess: {
    methods: ['majority_voting'],
    majority_voting: {
      n: 8,
      method: 'exact_match',
    },
  },
  evaluation: {
    answer_comparison: {
      method: 'semantic',
    },
  },
  rewrite: {
    method: 'difficulty_adjust',
  },
  translation: {
    language: 'english',
    model_path: null,
  },
}

export const useTaskStore = create<TaskState>()(
  devtools(
    (set, get) => ({
      config: DEFAULT_CONFIG,
      taskId: null,
      isGenerating: false,
      eventData: null,
      stepHistory: {},
      currentPhase: null,
      refineEventData: null,
      isRefining: false,
      generationCompleted: false,

      // 累加统计初始值
      accumulatedTokens: 0,
      timerStartTime: null,
      pausedDuration: 0,
      lastPauseTime: null,

      // Configuration 页面初始值
      hasAutoOpenedConfigModal: false,

      // SSE 连接状态初始值
      generationSSEConnected: false,
      refineSSEConnected: false,

      // SSE 连接追踪初始值
      hasConnectedGeneration: false,
      hasConnectedRefine: false,

      setTaskConfig: newConfig =>
        set(state => ({
          config: { ...state.config, ...newConfig },
        })),

      setTaskId: id => set({ taskId: id }),

      setIsGenerating: status => set({ isGenerating: status }),

      setEventData: data =>
        set(state => {
          const updates: Partial<TaskState> = {}

          const phaseId = data.phase?.id || 'unknown'

          if (data.step?.status === 'completed' && data.step?.usage?.tokens) {
            const newTokens = data.step.usage.tokens
            updates.accumulatedTokens = (state.accumulatedTokens || 0) + newTokens
          }

          if (phaseId === 'generation') {
            updates.eventData = data
            updates.currentPhase = 'generation'

            if (data.job_id && !state.taskId) {
              updates.taskId = data.job_id
            }

            if (data.status === 'generation_complete') {
              updates.generationCompleted = true
              updates.isGenerating = false
            }

            if (data.status === 'error' || data.status === 'failed') {
              updates.isGenerating = false
              get().pauseTimer()
            }
          } else if (phaseId === 'refinement') {
            updates.refineEventData = data
            updates.currentPhase = 'refine'

            if (data.status === 'completed') {
              updates.isRefining = false
            }

            if (data.status === 'error' || data.status === 'failed') {
              updates.isRefining = false
              get().pauseTimer()
            }
          }

          if (data.step) {
            const step = data.step
            const compositeKey = `${phaseId}:${step.id}`

            let stepStatus: 'pending' | 'loading' | 'success' | 'error'
            let stepMessage = step.message

            if (data.status === 'error' || data.status === 'failed') {
              stepStatus = 'error'
              if (data.error?.message) {
                stepMessage = data.error.message
              }
            } else if (step.status === 'completed') {
              stepStatus = 'success'
            } else if (step.status === 'running') {
              stepStatus = 'loading'
            } else if (step.result !== null) {
              stepStatus = 'success'
            } else if (step.progress !== null || step.usage !== null || step.message !== null) {
              stepStatus = 'loading'
            } else {
              stepStatus = 'pending'
            }

            updates.stepHistory = {
              ...state.stepHistory,
              [compositeKey]: {
                id: step.id,
                phaseId: phaseId,
                name: step.name,
                message: stepMessage,
                status: stepStatus,
                usage: step.usage,
                result: step.result,
              },
            }
          }

          return updates
        }),

      // 计算目标进度（基于 phase 的 current_step_index 和 total_steps）
      getTargetProgress: (eventData: SSEEventData | null): number => {
        if (!eventData) return 0

        if (eventData.status === 'error' || eventData.status === 'failed') {
          if (eventData.phase) {
            const { current_step_index, total_steps } = eventData.phase
            return Math.min((current_step_index / total_steps) * 100, 99)
          }
          return 0
        }

        if (
          eventData.status === 'generation_complete' ||
          eventData.status === 'complete' ||
          eventData.status === 'completed'
        ) {
          return 100
        }

        if (eventData.phase) {
          const { current_step_index, total_steps } = eventData.phase
          const rawProgress = (current_step_index / total_steps) * 100

          return Math.min(rawProgress, 99)
        }

        return 0
      },

      resetTask: () =>
        set({
          taskId: null,
          isGenerating: false,
          eventData: null,
          stepHistory: {},
          currentPhase: null,
          refineEventData: null,
          isRefining: false,
          generationCompleted: false,
          generationSSEConnected: false,
          refineSSEConnected: false,
          hasConnectedGeneration: false,
          hasConnectedRefine: false,
        }),

      // 新增 Actions
      setCurrentPhase: phase => set({ currentPhase: phase }),

      setRefineEventData: data => {
        useTaskStore.getState().setEventData(data)
      },

      setIsRefining: isRefining => set({ isRefining }),

      setGenerationCompleted: completed => set({ generationCompleted: completed }),

      startTimer: () =>
        set(state => {
          if (state.timerStartTime !== null) {
            return {}
          }
          const now = Date.now()
          return {
            timerStartTime: now,
            lastPauseTime: null,
          }
        }),

      pauseTimer: () =>
        set(state => {
          if (state.timerStartTime === null) {
            return {}
          }
          if (state.lastPauseTime !== null) {
            return {}
          }
          const now = Date.now()
          return {
            lastPauseTime: now,
          }
        }),

      resumeTimer: () =>
        set(state => {
          if (state.lastPauseTime === null) {
            return {}
          }
          const now = Date.now()
          const pauseDuration = now - state.lastPauseTime
          return {
            pausedDuration: state.pausedDuration + pauseDuration,
            lastPauseTime: null,
          }
        }),

      resetStats: () =>
        set({
          accumulatedTokens: 0,
          timerStartTime: null,
          pausedDuration: 0,
          lastPauseTime: null,
        }),

      getCurrentElapsedTime: () => {
        const state = get()
        if (state.timerStartTime === null) {
          return 0
        }

        const now = Date.now()
        let elapsed = now - state.timerStartTime - state.pausedDuration

        if (state.lastPauseTime !== null) {
          elapsed -= now - state.lastPauseTime
        }

        // 转换为秒
        return Math.max(0, elapsed / 1000)
      },

      setHasAutoOpenedConfigModal: hasOpened => set({ hasAutoOpenedConfigModal: hasOpened }),

      // SSE 连接状态相关方法
      setGenerationSSEConnected: connected => set({ generationSSEConnected: connected }),
      setRefineSSEConnected: connected => set({ refineSSEConnected: connected }),
      setHasConnectedGeneration: connected => set({ hasConnectedGeneration: connected }),
      setHasConnectedRefine: connected => set({ hasConnectedRefine: connected }),
    }),
    { name: 'TaskStore' },
  ),
)
