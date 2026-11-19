export type ModelType =
  | "efficientnet_b3"
  | "cnn_rf"
  | "cnn"
  | "cnn_dnn"
  | "cnn_tw3"
  | "multimodal_cnn"
  | "regression_multimodal"
  | "ensemble_cnn"
  | "multiscale_cnn_rf";

export interface ModelInfo {
  id: ModelType;
  display_name: string;
  description: string;
  requires_gender: boolean;
  supports_gradcam: boolean;
  checkpoint_available: boolean;
  is_experimental?: boolean;
  is_legacy?: boolean;
}

export interface ValidationDetails {
  is_valid: boolean;
  predicted_class: string;
  confidence: number;
  threshold: number;
  rejection_code?: string | null;
  rejection_message?: string | null;
  processing_time_ms?: number;
  probabilities?: Record<string, number>;
}

export interface PredictionResult {
  success?: boolean;
  error?: string;
  message?: string;
  model_type: string;
  bone_age_months: number;
  confidence: number;
  gender_used?: string | null;
  processing_time_ms: number;
  uncertainty_lower_months?: number | null;
  uncertainty_upper_months?: number | null;
  uncertainty_interval_months?: number | null;
  uncertainty_note?: string | null;
  inference_engine?: string;
  validation?: ValidationDetails;
}

export interface GradCamResult {
  success?: boolean;
  error?: string;
  message?: string;
  model_type: string;
  bone_age_months: number;
  confidence: number;
  method?: string;
  heatmap_base64: string;
  overlay_base64: string;
  processing_time_ms: number;
  validation?: ValidationDetails;
}

export interface MetricValues {
  mae: number;
  mse: number;
  rmse: number;
}

export interface ModelMetrics {
  model_type: string;
  display_name: string;
  metrics: MetricValues;
  num_samples: number;
}

export interface ComparisonData {
  models: ModelMetrics[];
  best_model: string;
  generated_at: string;
}

export interface EvaluationData {
  comparison: ComparisonData;
  plots: Record<string, string>;
  details?: Record<string, unknown>;
}
