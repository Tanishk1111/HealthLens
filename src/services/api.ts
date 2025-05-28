import axios from 'axios'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

export const api = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
})

export interface HealthCheckResponse {
  status: string
  services: {
    model_2d_handler_ready: boolean
    model_3d_handler_ready: boolean
    monai_model_handler_ready: boolean
    sonar_client_configured: boolean
  }
}

export interface Detection {
  class_name: string
  confidence: number
  bounding_box: number[] | 'N/A'
  segmentation_info: object | 'N/A'
}

export interface AnalysisResponse {
  scan_type_processed: string
  original_filename: string
  vision_analysis: {
    detections: Detection[]
    metadata: {
      model_used: string
      processing_time_seconds: number
      [key: string]: any
    }
  }
  sonar_query_sent_to_llm: string
  sonar_llm_explanation: string
}

export const healthCheck = async (): Promise<HealthCheckResponse> => {
  const response = await api.get<HealthCheckResponse>('/health')
  return response.data
}

export const analyzeScan = async (
  file: File,
  scanType: string,
  confidenceThreshold: number
): Promise<AnalysisResponse> => {
  const formData = new FormData()
  formData.append('file', file)
  formData.append('scan_type', scanType)
  formData.append('confidence_threshold', confidenceThreshold.toString())

  const response = await api.post<AnalysisResponse>('/analyze_scan/', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  })
  return response.data
}