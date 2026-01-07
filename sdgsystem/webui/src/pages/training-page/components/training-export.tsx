import {
  CloseOutlined,
  ExclamationCircleOutlined,
  ExportOutlined,
  FolderOpenOutlined,
  SyncOutlined,
} from '@ant-design/icons'
import { EventStreamContentType, fetchEventSource } from '@microsoft/fetch-event-source'
import { Button, Form, Input, Modal, message, Switch, Tag, Typography } from 'antd'
import { memo, useCallback, useEffect, useRef, useState } from 'react'
import { useAntdTheme } from '@/hooks/use-antd-theme'
import { useTrainingStore } from '@/store/use-training-store'

const { Text, Title } = Typography

// 常量：最大日志数量限制
const MAX_LOGS = 500

// Memo 化的日志项组件，避免不必要的重渲染
const LogItem = memo(({ log }: { log: string }) => (
  <div className="mb-1 whitespace-pre-wrap break-all">{log}</div>
))
LogItem.displayName = 'LogItem'

interface TrainingExportProps {
  jobId?: string | null
  onTaskComplete?: () => void
  onStatusChange?: (status: 'idle' | 'running' | 'completed' | 'cancelled' | 'error') => void
}

interface ExportFormValues {
  hfToken: string
  repoId: string
  private?: boolean
  commitMessage?: string
}

interface ExportResult {
  job_id: string
  repo_id: string
  repo_url: string
}

export default function TrainingExport({
  jobId,
  onTaskComplete,
  onStatusChange,
}: TrainingExportProps) {
  const token = useAntdTheme()
  const experimentName = useTrainingStore(state => state.trainingConfig.trainer.experiment_name)
  const method = useTrainingStore(state => state.trainingConfig.method)
  const modelPath = useTrainingStore(state => state.trainingConfig.model.path)

  const [status, setStatus] = useState<'idle' | 'running' | 'completed' | 'cancelled' | 'error'>(
    'idle',
  )
  const [_progress, setProgress] = useState(0)
  const [logs, setLogs] = useState<string[]>([])
  const [showCancelModal, setShowCancelModal] = useState(false)
  const [showExportModal, setShowExportModal] = useState(false)
  const [exportLoading, setExportLoading] = useState(false)
  const [exportResult, setExportResult] = useState<ExportResult | null>(null)
  const [wandbUrl, setWandbUrl] = useState<string | null>(null)
  const logContainerRef = useRef<HTMLDivElement | null>(null)
  const [autoScrollEnabled, setAutoScrollEnabled] = useState(true) // 仅在用户停留底部时自动滚动
  const sseControllerRef = useRef<AbortController | null>(null)
  const hasConnectedRef = useRef(false) // 防止重复连接

  // Stats
  const [_stats, setStats] = useState({
    lr: '--',
    loss: '--',
    stepTime: '--',
  })

  const handleCancelTask = async () => {
    if (!jobId) return

    try {
      // Call cancel API
      const res = await fetch(`/api/train/${jobId}/cancel`, {
        method: 'POST',
      })

      if (!res.ok) {
        throw new Error('Failed to cancel training task')
      }

      setStatus('cancelled')
      sseControllerRef.current?.abort()
      setShowCancelModal(false)
      setProgress(0)

      // Clear localStorage when task is cancelled
      localStorage.removeItem('training_job_state')

      // Notify parent component to clear jobId
      onTaskComplete?.()

      message.success('Task Cancelled Successfully')
    } catch (error) {
      const err = error as Error
      message.error(err.message || 'Failed to cancel task')
    }
  }

  const handleExport = async (values: ExportFormValues) => {
    if (!jobId) return

    setExportLoading(true)
    setExportResult(null) // 清空之前的结果

    try {
      const payload = {
        hf_token: values.hfToken,
        repo_id: values.repoId,
        private: values.private ?? false,
        commit_message: values.commitMessage || 'Upload model checkpoint',
      }

      const res = await fetch(`/api/train/${jobId}/upload`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
      })

      if (!res.ok) {
        const text = await res.text()
        throw new Error(text || 'Failed to export model')
      }

      const result: ExportResult = await res.json()
      setExportResult(result)
      message.success('Model exported successfully!')
    } catch (error) {
      const err = error as Error
      message.error(err.message || 'Failed to export model')
    } finally {
      setExportLoading(false)
    }
  }

  const handleLogScroll = () => {
    const container = logContainerRef.current
    if (!container) return
    // Check if user is near bottom (within 20px of bottom)
    const nearBottom =
      container.scrollHeight - container.scrollTop - container.clientHeight <= 20
    setAutoScrollEnabled(nearBottom)
  }

  useEffect(() => {
    if (!autoScrollEnabled) return
    const container = logContainerRef.current
    if (!container) return
    // Scroll to bottom when new logs are added
    container.scrollTo({ top: container.scrollHeight, behavior: 'smooth' })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [logs.length, autoScrollEnabled])

  useEffect(() => {
    return () => {
      sseControllerRef.current?.abort()
    }
  }, [])

  // Notify parent component when status changes
  useEffect(() => {
    onStatusChange?.(status)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [status])
  // Note: onStatusChange 有意排除在依赖之外，避免父组件重渲染时触发此 effect

  const unescapeLog = useCallback((str: string) => {
    try {
      // 尝试作为 JSON 字符串解析 (处理带引号的字符串: "...")
      // 这会自动处理 \uXXXX 转义
      const parsed = JSON.parse(str)
      if (typeof parsed === 'string') return parsed
      // 如果是对象或其他类型，根据需要处理，这里假设日志主要是文本
      if (typeof parsed === 'object') return JSON.stringify(parsed, null, 2)
      return String(parsed)
    } catch {
      // 如果不是有效的 JSON (例如没有引号的裸字符串)，手动处理 unicode 转义
      return str.replace(/\\u[\dA-F]{4}/gi, match =>
        String.fromCharCode(parseInt(match.replace(/\\u/g, ''), 16)),
      )
    }
  }, [])

  // Parse log for metrics
  const parseMetrics = useCallback((log: string) => {
    // Regex to capture progress percentage
    // Matches "Epoch X/Y:  25%"
    const progressMatch = log.match(/Epoch\s+\d+\/\d+:\s+(\d+)%/)
    if (progressMatch) {
      const percent = Number(progressMatch[1])
      if (!Number.isNaN(percent)) {
        setProgress(percent)
      }
    }

    // Regex to capture metrics
    // step:1 - train/loss:0.723 - train/lr(1e-3):0.0096 - train/time(s):2.81
    const lossMatch = log.match(/train\/loss:([\d.]+)/)
    const lrMatch = log.match(/train\/lr[^:]*:([\d.]+)/)
    const timeMatch = log.match(/train\/time\(s\):([\d.]+)/)

    if (lossMatch || lrMatch || timeMatch) {
      setStats(prev => ({
        lr: lrMatch ? Number(lrMatch[1]).toExponential(2) : prev.lr, // Use scientific notation for small LRs
        loss: lossMatch ? Number(lossMatch[1]).toFixed(4) : prev.loss,
        stepTime: timeMatch ? `${Number(timeMatch[1]).toFixed(2)}s` : prev.stepTime,
      }))
    }
  }, [])

  useEffect(() => {
    // 防止重复连接：如果没有 jobId 或已经连接，则不执行
    if (!jobId || hasConnectedRef.current) return

    // 标记为已连接
    hasConnectedRef.current = true

    sseControllerRef.current?.abort()
    setLogs([])
    setStatus('running')
    // Reset stats
    setStats({
      lr: '--',
      loss: '--',
      stepTime: '--',
    })

    const controller = new AbortController()
    sseControllerRef.current = controller

    // Using the start endpoint as implied by previous context,
    // assuming the creation (POST /api/train) and monitoring (SSE) are separate.
    fetchEventSource(`/api/train/${jobId}/start`, {
      method: 'POST',
      signal: controller.signal,
      headers: {
        'Content-Type': 'application/json',
      },
      async onopen(response) {
        const contentType = response.headers.get('content-type')
        if (
          response.ok &&
          (contentType === EventStreamContentType || contentType?.includes('text/event-stream'))
        ) {
          return
        }
        if (response.status >= 400 && response.status < 500) {
          throw new Error(`Connection failed: ${response.status}`)
        }
      },
      onmessage(msg) {
        if (!msg.data) return
        if (msg.event === 'log') {
          const logContent = unescapeLog(msg.data)
          // Log event content goes directly to the log component
          // 限制日志数量，避免内存占用过高（尾插法：最新的在底部）
          setLogs(prev => {
            const newLogs = [...prev, logContent]
            // Keep only the last MAX_LOGS entries
            return newLogs.length > MAX_LOGS ? newLogs.slice(-MAX_LOGS) : newLogs
          })
          // Parse metrics from the log line
          parseMetrics(logContent)

          // Extract WandB URL from log if present
          // Format: "wandb: ⭐️ View project at: https://wandb.ai/..."
          const wandbMatch = logContent.match(/https:\/\/wandb\.ai\/[^\s]+/)
          if (wandbMatch) {
            setWandbUrl(wandbMatch[0])
          }
        }
        if (msg.event === 'complete') {
          setStatus('completed')
          setProgress(100)
          setStats(prev => ({ ...prev, estTime: '0m' }))

          // Don't clear localStorage - allow user to see completed task after refresh
          // Don't clear jobId - user still needs it to export model

          controller.abort()
          message.success('Training completed')
        }
        if (msg.event === 'error') {
          setStatus('error')

          // Clear localStorage when training encounters an error
          localStorage.removeItem('training_job_state')

          // Notify parent component to clear jobId (can start new task)
          onTaskComplete?.()

          controller.abort()
          message.error(msg.data || 'Training failed with an error')
        }
      },
      onclose() {
        if (controller.signal.aborted) return

        // If server closes connection unexpectedly (not via 'complete' or 'error' events),
        // we should stop retrying to avoid duplicate connections and 400 errors
        if (status !== 'completed') {
          console.warn('SSE connection closed unexpectedly, stopping retries')
          controller.abort()
          setStatus('error')
          message.warning('Connection closed unexpectedly. Please check your training task status.')
        }
      },
      onerror(err) {
        if (controller.signal.aborted) return
        setStatus('error')

        // Clear localStorage when training encounters an error
        localStorage.removeItem('training_job_state')

        // Notify parent component to clear jobId
        onTaskComplete?.()

        // Don't throw to avoid re-throw loops if fetchEventSource retries aggressively
        // But here we want to stop
        controller.abort()

        // 改进错误提示，根据错误类型提供更详细的信息
        let errorMessage = 'Training log connection error'
        if (err instanceof Error) {
          const errMsg = err.message.toLowerCase()
          if (errMsg.includes('network') || errMsg.includes('fetch')) {
            errorMessage = 'Network connection failed. Please check your internet connection.'
          } else if (errMsg.includes('timeout')) {
            errorMessage = 'Connection timeout. Please try again.'
          } else if (errMsg.includes('abort')) {
            errorMessage = 'Connection was aborted.'
          } else {
            errorMessage = `Training error: ${err.message}`
          }
        }

        message.error(errorMessage, 5) // 显示5秒
        // Rethrow to stop retries
        throw err
      },
    })

    return () => {
      controller.abort()
      hasConnectedRef.current = false // 组件卸载时重置连接标记
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [jobId])
  // 注意：只依赖 jobId，仅在 jobId 改变时重新建立 SSE 连接
  //
  // 以下依赖被有意排除，避免不必要的重连：
  // - status: useEffect 内部会 setStatus('running')，如果 status 作为依赖会立即触发重连
  // - onTaskComplete: 父组件每次渲染都是新的函数引用，会导致重连
  // - parseMetrics/unescapeLog: 虽然是 useCallback 引用稳定，但只在回调中使用，无需作为依赖

  return (
    <div className="flex flex-col gap-4 pb-0" style={{ height: 'calc(100vh - 180px)' }}>
      {/* Top Header Section */}
      <div className="mx-auto w-full max-w-[1400px] flex-shrink-0 px-8">
        <div className="flex items-start justify-between">
          {/* Left: Task Info */}
          <div className="flex flex-col gap-3">
            <div className="flex items-center gap-3">
              <SyncOutlined spin={status === 'running'} style={{ fontSize: 20 }} />
              <span className="font-bold text-xl">{experimentName || 'Untitled Task'}</span>
              <Button
                danger
                icon={<CloseOutlined />}
                shape="round"
                size="small"
                onClick={() => setShowCancelModal(true)}
                disabled={status !== 'running'}
              >
                Cancel Task
              </Button>
              <Button
                icon={<ExportOutlined />}
                shape="round"
                size="small"
                disabled={!wandbUrl}
                onClick={() => {
                  if (wandbUrl) {
                    window.open(wandbUrl, '_blank', 'noopener,noreferrer')
                  }
                }}
                style={{
                  color: wandbUrl ? token.colorPrimary : undefined,
                  borderColor: wandbUrl ? token.colorPrimary : undefined,
                  background: 'transparent',
                }}
              >
                View in WandB
              </Button>
            </div>

            <div className="flex gap-2">
              <Tag
                bordered={false}
                style={{
                  background: token.colorFillContent,
                  border: `1px solid ${token.colorBorderSecondary}`,
                  marginRight: 0,
                  padding: '4px 12px',
                  display: 'flex',
                  alignItems: 'center',
                  fontSize: token.fontSize,
                }}
              >
                <span style={{ color: token.colorTextSecondary, marginRight: 4 }}>Mode</span>
                <span style={{ fontWeight: 600, color: token.colorText }}>
                  {(method || 'SFT').toUpperCase()}
                </span>
              </Tag>
              <Tag
                bordered={false}
                style={{
                  background: token.colorFillContent,
                  border: `1px solid ${token.colorBorderSecondary}`,
                  marginRight: 0,
                  padding: '4px 12px',
                  display: 'flex',
                  alignItems: 'center',
                  fontSize: token.fontSize,
                }}
              >
                <span style={{ color: token.colorTextSecondary, marginRight: 4 }}>Model</span>
                <span style={{ fontWeight: 600, color: token.colorText }}>
                  {modelPath.split('/').pop() || modelPath}
                </span>
              </Tag>
              <Tag
                bordered={false}
                style={{
                  background: token.colorFillContent,
                  border: `1px solid ${token.colorBorderSecondary}`,
                  marginRight: 0,
                  padding: '4px 12px',
                  display: 'flex',
                  alignItems: 'center',
                  fontSize: token.fontSize,
                }}
              >
                <span style={{ color: token.colorTextSecondary, marginRight: 4 }}>Dataset</span>
                <span style={{ fontWeight: 600, color: token.colorText }}>Custom Dataset</span>
              </Tag>
            </div>
          </div>

          {/* Right: Export Button */}
          <div className="flex items-center gap-4">
            <Button
              icon={<FolderOpenOutlined />}
              className="h-12 px-6"
              disabled={status !== 'completed'}
              style={{
                color: status === 'completed' ? token.colorPrimary : undefined,
                borderColor: status === 'completed' ? token.colorPrimaryBorder : undefined,
                background: status === 'completed' ? token.colorPrimaryBg : undefined,
              }}
              onClick={() => setShowExportModal(true)}
            >
              Export Model
            </Button>
          </div>
        </div>
      </div>

      {/* Terminal / Logs - Flex 1 to take remaining space */}
      <div className="mx-auto flex min-h-0 w-full max-w-[1400px] flex-1 flex-col px-8">
        <div
          className="flex-1 overflow-hidden overflow-y-auto rounded-lg p-6 font-mono text-sm"
          style={{
            backgroundColor: '#0F0524', // Keeping this for terminal look
            color: '#E0E0E0',
            border: `1px solid ${token.colorBorderSecondary}`,
          }}
          ref={logContainerRef}
          onScroll={handleLogScroll}
        >
          {status === 'running' && (
            <div className="mb-2 animate-pulse font-bold text-green-400">&gt; Processing...</div>
          )}
          {!jobId && (
            <div className="text-[#9CA3AF] text-sm">
              Please create a training task in Training Config first to view logs
            </div>
          )}
          {logs.map((log, index) => (
            <LogItem key={`log-${index}`} log={log} />
          ))}
        </div>
      </div>

      {/* Cancel Confirmation Modal */}
      <Modal
        open={showCancelModal}
        onCancel={() => setShowCancelModal(false)}
        footer={null}
        width={400}
        centered
        closeIcon={false}
        className="overflow-hidden rounded-xl"
      >
        <div className="flex flex-col items-center pt-4 text-center">
          <div
            className="mb-4 flex h-16 w-16 items-center justify-center rounded-full"
            style={{ background: token.colorErrorBg }}
          >
            <ExclamationCircleOutlined className="text-3xl" style={{ color: token.colorError }} />
          </div>
          <Title level={4}>Cancel Task?</Title>
          <Text className="mb-8 px-4" style={{ color: token.colorTextSecondary }}>
            Are you sure you want to cancel the current task? All progress will be lost.
          </Text>
          <div className="flex w-full gap-3">
            <Button danger type="primary" block size="large" onClick={handleCancelTask}>
              Cancel Task
            </Button>
            <Button block size="large" onClick={() => setShowCancelModal(false)}>
              Close
            </Button>
          </div>
        </div>
      </Modal>

      {/* Export Model Modal */}
      <Modal
        title={
          <div className="flex items-center gap-2 text-lg">
            <div
              className="rounded p-1"
              style={{ background: token.colorPrimaryBg, color: token.colorPrimary }}
            >
              <FolderOpenOutlined />
            </div>
            Export Model
          </div>
        }
        open={showExportModal}
        onCancel={() => {
          setShowExportModal(false)
          setExportResult(null) // 关闭弹窗时重置结果
        }}
        footer={null}
        width={500}
        centered
      >
        <Form layout="vertical" onFinish={handleExport} className="mt-6">
          <Form.Item
            label="Hugging Face Token"
            name="hfToken"
            required
            rules={[{ required: true, message: 'Please enter Hugging Face Token' }]}
          >
            <Input placeholder="hf_..." size="large" disabled={exportLoading} />
          </Form.Item>

          <Form.Item
            label="Repository ID"
            name="repoId"
            required
            rules={[{ required: true, message: 'Please enter Repository ID' }]}
          >
            <Input placeholder="username/model-name" size="large" disabled={exportLoading} />
          </Form.Item>

          <Form.Item label="Private Repository" name="private" valuePropName="checked">
            <Switch disabled={exportLoading} />
          </Form.Item>

          <Form.Item label="Commit Message" name="commitMessage">
            <Input placeholder="Upload model checkpoint" size="large" disabled={exportLoading} />
          </Form.Item>

          {/* 显示导出结果 */}
          {exportResult && (
            <div
              className="mb-4 rounded-lg p-4"
              style={{
                background: token.colorPrimaryBg,
                border: `1px solid ${token.colorPrimaryBorder}`,
              }}
            >
              <div className="mb-2 flex items-center gap-2">
                <Text strong style={{ color: token.colorPrimary }}>
                  Model Exported Successfully!
                </Text>
              </div>
              <div className="mb-2">
                <Text style={{ color: token.colorTextSecondary, fontSize: '12px' }}>
                  Repository:
                </Text>
                <Text
                  strong
                  style={{ color: token.colorText, fontSize: '14px', marginLeft: '8px' }}
                >
                  {exportResult.repo_id}
                </Text>
              </div>
              <div>
                <Button
                  type="link"
                  icon={<ExportOutlined />}
                  size="small"
                  onClick={() =>
                    window.open(exportResult.repo_url, '_blank', 'noopener,noreferrer')
                  }
                  style={{ padding: 0, height: 'auto' }}
                >
                  Visit on Hugging Face
                </Button>
              </div>
            </div>
          )}

          <div className="mt-8 flex justify-end gap-3">
            {exportResult && (
              <Button
                size="large"
                style={{ width: '120px' }}
                onClick={() => {
                  setShowExportModal(false)
                  setExportResult(null)
                }}
              >
                Close
              </Button>
            )}
            <Button
              type="primary"
              htmlType="submit"
              size="large"
              loading={exportLoading}
              disabled={!!exportResult} // 导出成功后禁用按钮
              style={{ width: '120px' }}
            >
              Export
            </Button>
          </div>
        </Form>
      </Modal>
    </div>
  )
}
