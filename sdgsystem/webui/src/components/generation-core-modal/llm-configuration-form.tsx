import { Form, Input, message, Select } from 'antd'
import { forwardRef, useEffect, useImperativeHandle } from 'react'
import { useAntdTheme } from '@/hooks/use-antd-theme'
import { useTaskStore } from '@/store/use-task-store'

const PROVIDER_OPTIONS = [
  { value: 'openai', label: 'OpenAI' },
  { value: 'deepseek', label: 'Deepseek' },
  { value: 'qwen', label: 'Qwen' },
  { value: 'ollama', label: 'Ollama' },
]

export interface FormRef {
  save: () => Promise<void>
}

interface LLMConfigurationFormProps {
  open: boolean
}

export const LLMConfigurationForm = forwardRef<FormRef, LLMConfigurationFormProps>(
  ({ open }, ref) => {
    const [form] = Form.useForm()
    const token = useAntdTheme()
    const { config, setTaskConfig } = useTaskStore()

    useEffect(() => {
      if (open) {
        form.setFieldsValue({
          provider: config.llm.provider,
          model: config.llm.model,
          apiKey: config.llm.api_key,
          baseUrl: config.llm.base_url,
        })
      }
    }, [open, config.llm, form])

    const handleValuesChange = (_changedValues: unknown, allValues: Record<string, unknown>) => {
      setTaskConfig({
        llm: {
          provider: (allValues.provider as string) || '',
          model: (allValues.model as string) || '',
          api_key: (allValues.apiKey as string) || '',
          base_url: (allValues.baseUrl as string) || '',
        },
      })
    }

    const handleSave = async () => {
      const values = await form.validateFields()
      setTaskConfig({
        llm: {
          provider: values.provider,
          model: values.model,
          api_key: values.apiKey,
          base_url: values.baseUrl,
        },
      })
      message.success('LLM Configuration saved successfully')
    }

    useImperativeHandle(ref, () => ({
      save: handleSave,
    }))

    return (
      <div className="flex h-full flex-col">
        <Form form={form} layout="vertical" className="flex-1" onValuesChange={handleValuesChange}>
          <Form.Item
            name="provider"
            label="LLM Provider"
            rules={[{ required: true, message: 'Please select LLM Provider' }]}
          >
            <Select placeholder="Select provider" options={PROVIDER_OPTIONS} size="large" />
          </Form.Item>

          <Form.Item
            name="model"
            label="LLM Model"
            rules={[{ required: true, message: 'Please enter LLM Model' }]}
          >
            <Input placeholder="gpt-4o-mini" size="large" />
          </Form.Item>
          <div
            style={{
              marginTop: -token.marginSM,
              marginBottom: token.margin,
              fontSize: token.fontSizeSM,
              color: token.colorTextTertiary,
            }}
          >
            e.g., gpt-4o-mini, claude-3-opus, gemini-pro.
          </div>

          <Form.Item
            name="apiKey"
            label="API Key"
            rules={[{ required: true, message: 'Please enter API Key' }]}
          >
            <Input.Password placeholder="API Key" size="large" />
          </Form.Item>

          <Form.Item
            name="baseUrl"
            label={
              <>
                Base URL{' '}
                <span style={{ color: token.colorTextTertiary, fontWeight: 'normal' }}>
                  (Optional)
                </span>
              </>
            }
          >
            <div style={{ marginTop: -token.marginXS }}>
              <div
                style={{
                  fontSize: token.fontSizeSM,
                  color: token.colorTextTertiary,
                  marginBottom: token.marginXS,
                }}
              >
                Custom API endpoint address.
              </div>
              <Input placeholder="https://api.openai.com/v1" size="large" />
            </div>
          </Form.Item>
        </Form>
      </div>
    )
  },
)

LLMConfigurationForm.displayName = 'LLMConfigurationForm'
