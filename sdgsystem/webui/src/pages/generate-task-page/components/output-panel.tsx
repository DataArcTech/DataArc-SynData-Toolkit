import { DownloadOutlined, SearchOutlined } from '@ant-design/icons'
import { Button, Card, Modal, message, Skeleton, Space, Table } from 'antd'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { taskApi } from '@/api/task'
import { useAntdTheme } from '@/hooks/use-antd-theme'
import { useTaskStore } from '@/store/use-task-store'

type TabType = 'raw' | 'solved' | 'learnable' | 'unsolved'

interface DataItem {
  key: string
  input: string
  output: string
}

export function OutputPanel() {
  const token = useAntdTheme()
  const { eventData, refineEventData, taskId } = useTaskStore()
  const [activeTab, setActiveTab] = useState<TabType>('raw')
  const [data, setData] = useState<Record<TabType, DataItem[]>>({
    raw: [],
    solved: [],
    learnable: [],
    unsolved: [],
  })
  const [loading, setLoading] = useState(false)
  const [modalVisible, setModalVisible] = useState(false)
  const [selectedItem, setSelectedItem] = useState<DataItem | null>(null)

  const loadingTypesRef = useRef<Set<TabType>>(new Set())
  const loadedTypesRef = useRef<Set<TabType>>(new Set())

  const generationCounts = eventData?.output?.counts || {
    raw: null,
    solved: null,
    learnable: null,
    unsolved: null,
  }

  const refineCounts = refineEventData?.output?.counts || {
    raw: null,
    solved: null,
    learnable: null,
    unsolved: null,
  }

  const counts = useMemo(
    () => ({
      raw: generationCounts.raw,
      solved: refineCounts.solved || generationCounts.solved,
      learnable: refineCounts.learnable || generationCounts.learnable,
      unsolved: refineCounts.unsolved || generationCounts.unsolved,
    }),
    [
      generationCounts.raw,
      refineCounts.solved,
      generationCounts.solved,
      refineCounts.learnable,
      generationCounts.learnable,
      refineCounts.unsolved,
      generationCounts.unsolved,
    ],
  )

  const hasAnyData = Object.values(counts).some(count => count !== null)

  const parseJsonl = useCallback((content: string): DataItem[] => {
    const lines = content.trim().split('\n').filter(Boolean)
    return lines.map((line, index) => {
      try {
        const parsed = JSON.parse(line)
        return {
          key: String(index + 1),
          input: parsed.input || '',
          output: parsed.output || '',
        }
      } catch (_e) {
        return {
          key: String(index + 1),
          input: 'Parse error',
          output: 'Parse error',
        }
      }
    })
  }, [])

  const loadData = useCallback(
    async (fileType: TabType, silent = false) => {
      if (!taskId || !counts[fileType]) {
        return
      }

      if (loadingTypesRef.current.has(fileType)) {
        return
      }

      if (loadedTypesRef.current.has(fileType)) {
        return
      }

      loadingTypesRef.current.add(fileType)
      if (!silent) {
        setLoading(true)
      }
      try {
        const content = await taskApi.downloadDataset(taskId, fileType)
        const parsedData = parseJsonl(content)
        setData(prev => ({ ...prev, [fileType]: parsedData }))
        loadedTypesRef.current.add(fileType)
      } catch (_error) {
        if (!silent) {
          message.error(`Failed to load ${fileType} data`)
        }
      } finally {
        if (!silent) {
          setLoading(false)
        }
        loadingTypesRef.current.delete(fileType)
      }
    },
    [taskId, counts, parseJsonl],
  )

  useEffect(() => {
    loadData(activeTab)
  }, [activeTab, loadData])

  useEffect(() => {
    if (counts.raw) {
      loadData('raw')
    }
  }, [counts.raw, loadData])

  useEffect(() => {
    if (refineEventData?.status === 'completed') {
      const typesToLoad: TabType[] = ['solved', 'learnable', 'unsolved']
      typesToLoad.forEach(type => {
        const count = counts[type]
        if (count != null && count > 0) {
          loadData(type, true)
        }
      })
    }
  }, [refineEventData?.status, counts, loadData])

  const handleDownload = async () => {
    if (!taskId || !counts[activeTab]) {
      message.warning('No data available for download')
      return
    }

    try {
      const content = await taskApi.downloadDataset(taskId, activeTab)
      const blob = new Blob([content], { type: 'application/jsonl' })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `${eventData?.task_name || 'dataset'}_${activeTab}.jsonl`
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      URL.revokeObjectURL(url)
      message.success('Download successful')
    } catch (_error) {
      message.error('Download failed')
    }
  }

  const handleRowClick = (record: DataItem) => {
    setSelectedItem(record)
    setModalVisible(true)
  }

  const columns = [
    {
      title: 'Input',
      dataIndex: 'input',
      key: 'input',
      width: '40%',
      ellipsis: true,
      filterIcon: <SearchOutlined />,
    },
    {
      title: 'Output',
      dataIndex: 'output',
      key: 'output',
      width: '60%',
      ellipsis: true,
      filterIcon: <SearchOutlined />,
    },
  ]

  const tabs: { key: TabType; label: string }[] = useMemo(() => {
    const allTabs = [
      { key: 'raw' as TabType, label: `Raw${counts.raw ? ` (${counts.raw})` : ''}`, show: true },
      {
        key: 'solved' as TabType,
        label: `Solved${counts.solved ? ` (${counts.solved})` : ''}`,
        show: !!counts.solved && data.solved.length > 0,
      },
      {
        key: 'learnable' as TabType,
        label: `Learnable${counts.learnable ? ` (${counts.learnable})` : ''}`,
        show: !!counts.learnable && data.learnable.length > 0,
      },
      {
        key: 'unsolved' as TabType,
        label: `Unsolved${counts.unsolved ? ` (${counts.unsolved})` : ''}`,
        show: !!counts.unsolved && data.unsolved.length > 0,
      },
    ]
    return allTabs.filter(tab => tab.show).map(({ key, label }) => ({ key, label }))
  }, [counts, data])

  return (
    <div className="col-span-8 h-[76dvh]">
      <Card
        className="h-full min-h-[600px] shadow-sm"
        style={{ borderColor: token.colorBorderSecondary }}
        styles={{ body: { padding: 24 } }}
      >
        {!hasAnyData ? (
          <div className="flex flex-col gap-6">
            <div className="flex items-center justify-between">
              <Skeleton.Button active size="large" style={{ width: 300 }} />
              <Skeleton.Button active size="large" style={{ width: 150 }} />
            </div>
            <Skeleton active paragraph={{ rows: 10 }} />
          </div>
        ) : (
          <div className="flex flex-col gap-6">
            <div className="flex items-center justify-between">
              <Space size="large" role="tablist">
                {tabs.map(tab => (
                  <div
                    key={tab.key}
                    role="tab"
                    tabIndex={0}
                    aria-selected={activeTab === tab.key}
                    className="cursor-pointer rounded-md px-3 py-1 font-medium text-base transition-colors"
                    style={{
                      color: activeTab === tab.key ? token.colorPrimary : token.colorTextSecondary,
                      backgroundColor: activeTab === tab.key ? token.colorPrimaryBg : 'transparent',
                      outline: 'none',
                      opacity: counts[tab.key] ? 1 : 0.5,
                    }}
                    onClick={() => counts[tab.key] && setActiveTab(tab.key)}
                    onKeyDown={e => {
                      if ((e.key === 'Enter' || e.key === ' ') && counts[tab.key]) {
                        setActiveTab(tab.key)
                      }
                    }}
                  >
                    {tab.label}
                  </div>
                ))}
              </Space>
              <Button
                icon={<DownloadOutlined />}
                onClick={handleDownload}
                disabled={!counts[activeTab]}
                style={{
                  color: token.colorPrimary,
                  borderColor: token.colorPrimary,
                }}
              >
                Download Data
              </Button>
            </div>

            <Table
              columns={columns}
              dataSource={data[activeTab]}
              loading={loading}
              pagination={{
                pageSize: 10,
                showSizeChanger: true,
                showTotal: total => `Total ${total} items`,
              }}
              rowKey="key"
              scroll={{ y: 500 }}
              bordered
              showSorterTooltip={false}
              onRow={record => ({
                onClick: () => handleRowClick(record),
                style: { cursor: 'pointer' },
              })}
            />
          </div>
        )}
      </Card>

      <Modal
        title="Data Detail"
        open={modalVisible}
        onCancel={() => setModalVisible(false)}
        footer={null}
        width={800}
        style={{ maxHeight: '80vh' }}
      >
        {selectedItem && (
          <div className="flex flex-col gap-4">
            <div>
              <div className="mb-2 font-semibold text-base" style={{ color: token.colorText }}>
                Input
              </div>
              <div
                className="rounded p-3"
                style={{
                  background: token.colorFillTertiary,
                  color: token.colorText,
                  whiteSpace: 'pre-wrap',
                  wordBreak: 'break-word',
                  maxHeight: '300px',
                  overflowY: 'auto',
                }}
              >
                {selectedItem.input}
              </div>
            </div>

            <div>
              <div className="mb-2 font-semibold text-base" style={{ color: token.colorText }}>
                Output
              </div>
              <div
                className="rounded p-3"
                style={{
                  background: token.colorFillTertiary,
                  color: token.colorText,
                  whiteSpace: 'pre-wrap',
                  wordBreak: 'break-word',
                  maxHeight: '300px',
                  overflowY: 'auto',
                }}
              >
                {selectedItem.output}
              </div>
            </div>
          </div>
        )}
      </Modal>
    </div>
  )
}
