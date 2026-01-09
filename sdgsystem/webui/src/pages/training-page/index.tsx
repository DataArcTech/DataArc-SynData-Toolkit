import { message, Segmented, Tooltip } from 'antd'
import { useEffect, useState } from 'react'
import { useAntdTheme } from '@/hooks/use-antd-theme'
import TrainingConfig from './components/training-config'
import TrainingExport from './components/training-export'

type TabKey = 'config' | 'export'

export default function TrainingPage() {
  const token = useAntdTheme()
  const [activeTab, setActiveTab] = useState<TabKey>('config')
  const [jobId, setJobId] = useState<string | null>(null)
  const [trainingStatus, setTrainingStatus] = useState<
    'idle' | 'running' | 'completed' | 'cancelled' | 'error'
  >('idle')

  // Restore job_id from localStorage on page load
  useEffect(() => {
    const storedJobState = localStorage.getItem('training_job_state')
    if (storedJobState) {
      try {
        const jobState = JSON.parse(storedJobState)
        setJobId(jobState.jobId)

        // Restore training status if available
        if (jobState.status) {
          setTrainingStatus(jobState.status)
        }

        // Auto-switch to export tab if task is running or completed
        // For error/cancelled tasks, stay on config tab
        if (jobState.status === 'running') {
          setActiveTab('export')
          message.info('Reconnecting to training task...')
        } else if (jobState.status === 'completed') {
          setActiveTab('export')
          message.success(
            `Training task completed: ${jobState.experimentName || 'Untitled'}. You can now export the model.`,
            5,
          )
        } else {
          // error or cancelled - stay on config tab
          const statusText: Record<string, string> = {
            error: 'failed',
            cancelled: 'cancelled',
          }
          const displayStatus = statusText[jobState.status] || 'previous'

          message.info(
            `Found ${displayStatus} task: ${jobState.experimentName || 'Untitled'}. Switch to "Training & Export" tab to view details.`,
            5,
          )
        }
      } catch (error) {
        console.error('Failed to parse stored job state:', error)
        localStorage.removeItem('training_job_state')
      }
    }
  }, [])

  // Listen to storage events for multi-tab synchronization
  useEffect(() => {
    const handleStorageChange = (e: StorageEvent) => {
      if (e.key === 'training_job_state') {
        if (e.newValue === null) {
          // Other tab cleared the jobId
          setJobId(null)
          // Don't auto-switch tab - let user stay on current page
        } else {
          try {
            const jobState = JSON.parse(e.newValue)
            setJobId(jobState.jobId)

            // Auto-switch to export tab if task is running or completed
            if (jobState.status === 'running' || jobState.status === 'completed') {
              setActiveTab('export')
            }
          } catch (error) {
            console.error('Failed to parse storage event:', error)
          }
        }
      }
    }

    window.addEventListener('storage', handleStorageChange)
    return () => window.removeEventListener('storage', handleStorageChange)
  }, [])

  const isTaskRunning = trainingStatus === 'running'

  const segmentedOptions = [
    {
      value: 'config' as TabKey,
      label: 'Training Config',
      disabled: isTaskRunning, // Disable when task is running
    },
    {
      value: 'export' as TabKey,
      label: 'Training & Export',
    },
  ]

  return (
    <div className="flex flex-col" style={{ background: token.colorBgLayout, minHeight: '100%' }}>
      <div className="w-full pt-8">
        <div style={{ marginBottom: token.marginLG }} className="mx-auto w-[50%] max-w-[640px]">
          <Tooltip
            title={
              isTaskRunning
                ? 'A task is currently in progress. To create a new task, please wait for completion or cancel the current task'
                : ''
            }
          >
            <Segmented
              value={activeTab}
              onChange={value => setActiveTab(value as TabKey)}
              options={segmentedOptions}
              size="large"
              block
              style={{
                background: token.colorFillQuaternary,
                padding: token.paddingXXS,
                borderRadius: token.borderRadiusLG,
              }}
            />
          </Tooltip>
        </div>

        <div style={{ display: activeTab === 'config' ? 'block' : 'none' }}>
          <div className="mx-auto w-full" style={{ maxWidth: 1280 }}>
            <TrainingConfig
              onStartTraining={id => {
                setJobId(id)
                setActiveTab('export')
                setTrainingStatus('running')
              }}
            />
          </div>
        </div>

        <div style={{ display: activeTab === 'export' ? 'block' : 'none' }}>
          {/* Full width container for Export */}
          <TrainingExport
            jobId={jobId}
            onTaskComplete={() => {
              setJobId(null)
              setTrainingStatus('idle')
            }}
            onStatusChange={status => {
              setTrainingStatus(status)
            }}
          />
        </div>
      </div>
    </div>
  )
}
