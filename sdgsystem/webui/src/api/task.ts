import type { TaskConfig } from '@/store/use-task-store'
import { request } from '@/utils/request'

export interface Task {
  id: string
  name: string
  description: string
  status: 'pending' | 'generating' | 'completed' | 'failed'
  createdAt: string
}

export interface CreateTaskParams {
  taskName: string
  taskDescription: string
  dataCount: number
  model: string
}

export interface CreateJobResponse {
  job_id: string
}

export interface CreateJobRequest {
  config: TaskConfig
  files?: File[]
}

export const taskApi = {
  // 创建任务
  createTask: (data: CreateTaskParams) => {
    return request.post<{ taskId: string }>('/tasks', data)
  },

  // 获取任务详情
  getTask: (taskId: string) => {
    return request.get<Task>(`/tasks/${taskId}`)
  },

  // 获取任务列表
  getTasks: () => {
    return request.get<Task[]>('/tasks')
  },

  // 创建 job
  createJob: (data: CreateJobRequest) => {
    const formData = new FormData()

    const configStr = JSON.stringify(data.config)
    formData.append('config', configStr)

    if (data.files && data.files.length > 0) {
      data.files.forEach(file => {
        formData.append('files', file)
      })
    }

    return request.post<CreateJobResponse>('/sdg', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    })
  },

  // 下载数据文件
  downloadDataset: (jobId: string, fileType: 'raw' | 'solved' | 'learnable' | 'unsolved') => {
    return request.get<string>(`/sdg/${jobId}/download/${fileType}`)
  },
}
