import { useCallback, useEffect, useState } from "react";
import { fetchModels, generateGradCam, predictBoneAge } from "../api/client";
import GradCamViewer from "../components/GradCamViewer";
import ImageUpload from "../components/ImageUpload";
import ModelSelector from "../components/ModelSelector";
import PredictionResultCard from "../components/PredictionResult";
import type { GradCamResult, ModelInfo, ModelType, PredictionResult, ValidationDetails } from "../types";

export default function PredictPage() {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [selectedModel, setSelectedModel] = useState<ModelType>("efficientnet_b3");
  const [gender, setGender] = useState("male");
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);

  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [gradcam, setGradcam] = useState<GradCamResult | null>(null);
  const [showHeatmapOnly, setShowHeatmapOnly] = useState(false);

  const [predictLoading, setPredictLoading] = useState(false);
  const [gradcamLoading, setGradcamLoading] = useState(false);
  const [predictError, setPredictError] = useState<string | null>(null);
  const [gradcamError, setGradcamError] = useState<string | null>(null);
  const [validationFailure, setValidationFailure] = useState<ValidationDetails | null>(null);

  useEffect(() => {
    fetchModels()
      .then((res) => {
        setModels(res.models);
        // Default to efficientnet_b3 if available, else backend default
        const eff = res.models.find((m) => m.id === "efficientnet_b3");
        if (eff && eff.checkpoint_available) {
          setSelectedModel("efficientnet_b3");
        } else {
          setSelectedModel(res.default_model as ModelType);
        }
      })
      .catch(() => setPredictError("Unable to connect to API"));
  }, []);

  const handleFileSelect = useCallback((f: File, url: string) => {
    setFile(f);
    setPreviewUrl(url);
    setPrediction(null);
    setGradcam(null);
    setPredictError(null);
    setGradcamError(null);
    setValidationFailure(null);
  }, []);

  const handleClearImage = useCallback(() => {
    setFile(null);
    setPreviewUrl(null);
    setPrediction(null);
    setGradcam(null);
    setPredictError(null);
    setGradcamError(null);
    setValidationFailure(null);
  }, []);

  const currentModel = models.find((m) => m.id === selectedModel);

  const handlePredict = async () => {
    if (!file) return;
    setPredictLoading(true);
    setPredictError(null);
    setValidationFailure(null);
    try {
      const genderParam = currentModel?.requires_gender ? gender : undefined;
      const result = await predictBoneAge(file, selectedModel, genderParam);
      setPrediction(result);
    } catch (err: unknown) {
      setPrediction(null); // Never display bone age prediction on rejected images
      const data =
        err && typeof err === "object" && "response" in err
          ? (
              err as {
                response?: {
                  data?: {
                    detail?: string;
                    message?: string;
                    error?: string;
                    validation?: ValidationDetails;
                  };
                };
              }
            ).response?.data
          : null;
      setValidationFailure(data?.validation ?? null);
      const msg = data?.message || data?.detail || "Prediction failed";
      setPredictError(msg);
    } finally {
      setPredictLoading(false);
    }
  };

  const handleGradCam = async () => {
    if (!file) return;
    setGradcamLoading(true);
    setGradcamError(null);
    try {
      const genderParam = currentModel?.requires_gender ? gender : undefined;
      const result = await generateGradCam(file, selectedModel, genderParam);
      setGradcam(result);
    } catch (err: unknown) {
      setGradcam(null); // Never display Grad-CAM on rejected images
      const data =
        err && typeof err === "object" && "response" in err
          ? (
              err as {
                response?: {
                  data?: {
                    detail?: string;
                    message?: string;
                    error?: string;
                    validation?: ValidationDetails;
                  };
                };
              }
            ).response?.data
          : null;
      setValidationFailure(data?.validation ?? null);
      const msg = data?.message || data?.detail || "Grad-CAM generation failed";
      setGradcamError(msg);
    } finally {
      setGradcamLoading(false);
    }
  };

  return (
    <div className="grid gap-6 lg:grid-cols-2">
      <div className="space-y-6">
        <div className="card">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold text-slate-800">
              Upload Pediatric Hand X-Ray
            </h2>
            {file && (
              <button
                type="button"
                onClick={handleClearImage}
                className="text-xs font-medium text-slate-500 hover:text-red-600 transition"
              >
                Clear Image
              </button>
            )}
          </div>
          <ImageUpload
            onFileSelect={handleFileSelect}
            disabled={predictLoading || gradcamLoading}
          />
          {previewUrl && (
            <div className="mt-4 overflow-hidden rounded-lg border border-slate-200">
              <img
                src={previewUrl}
                alt="X-ray preview"
                className="max-h-64 w-full object-contain bg-black"
              />
            </div>
          )}
        </div>

        <div className="card">
          <ModelSelector
            models={
              models.length
                ? models
                : [
                    {
                      id: "cnn_rf",
                      display_name: "CNN + Random Forest",
                      description: "ResNet-18 feature extractor + RF regressor (Image-Only, Best Val MAE)",
                      requires_gender: false,
                      supports_gradcam: true,
                      checkpoint_available: true,
                      is_experimental: false,
                    },
                    {
                      id: "cnn",
                      display_name: "CNN Baseline",
                      description: "ResNet-18 backbone with linear regression head (Image-Only)",
                      requires_gender: false,
                      supports_gradcam: true,
                      checkpoint_available: true,
                      is_experimental: false,
                    },
                    {
                      id: "cnn_dnn",
                      display_name: "CNN + DNN",
                      description: "ResNet-18 with 3-layer deep regression head (Image-Only)",
                      requires_gender: false,
                      supports_gradcam: true,
                      checkpoint_available: true,
                      is_experimental: false,
                    },
                    {
                      id: "cnn_tw3",
                      display_name: "CNN + TW3 Proxy",
                      description: "CNN features fused with image-derived TW3 proxy features (Image-Only)",
                      requires_gender: false,
                      supports_gradcam: true,
                      checkpoint_available: true,
                      is_experimental: false,
                    },
                    {
                      id: "multimodal_cnn",
                      display_name: "Multimodal CNN (Experimental)",
                      description: "Experimental research model combining image features with biological sex",
                      requires_gender: true,
                      supports_gradcam: true,
                      checkpoint_available: true,
                      is_experimental: true,
                    },
                  ]
            }
            selected={selectedModel}
            onChange={setSelectedModel}
            gender={gender}
            onGenderChange={setGender}
            disabled={predictLoading || gradcamLoading}
          />

          <div className="mt-6 flex flex-wrap gap-3">
            <button
              type="button"
              className="btn-primary"
              onClick={handlePredict}
              disabled={!file || predictLoading}
            >
              {predictLoading ? "Validating & Predicting…" : "Predict Bone Age"}
            </button>
            <button
              type="button"
              className="btn-secondary"
              onClick={handleGradCam}
              disabled={!file || gradcamLoading}
            >
              {gradcamLoading ? "Validating & Generating…" : "Generate Grad-CAM"}
            </button>
          </div>
        </div>
      </div>

      <div className="space-y-6">
        <PredictionResultCard
          result={prediction}
          loading={predictLoading}
          error={predictError}
          validationFailure={validationFailure}
          onClear={handleClearImage}
        />
        {/* Only show Grad-CAM viewer if no validation error and prediction or gradcam is active */}
        {!predictError && (
          <GradCamViewer
            overlayBase64={gradcam?.overlay_base64 ?? null}
            heatmapBase64={gradcam?.heatmap_base64 ?? null}
            loading={gradcamLoading}
            error={gradcamError}
            showHeatmap={showHeatmapOnly}
            onToggleView={() => setShowHeatmapOnly((v) => !v)}
          />
        )}
      </div>
    </div>
  );
}
