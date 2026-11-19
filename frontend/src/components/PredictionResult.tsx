import type { PredictionResult, ValidationDetails } from "../types";

interface PredictionResultCardProps {
  result: PredictionResult | null;
  loading?: boolean;
  error?: string | null;
  validationFailure?: ValidationDetails | null;
  onClear?: () => void;
}

function yearsMonths(totalMonths: number): string {
  const years = Math.floor(totalMonths / 12);
  const months = Math.round(totalMonths % 12);
  return `${years}y ${months}m`;
}

export default function PredictionResultCard({
  result,
  loading,
  error,
  validationFailure,
  onClear,
}: PredictionResultCardProps) {
  if (loading) {
    return (
      <div className="card animate-pulse space-y-4">
        <div className="h-4 w-32 rounded bg-slate-200" />
        <div className="h-12 w-48 rounded bg-slate-200" />
        <div className="h-3 w-full rounded bg-slate-100" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="card border-red-200 bg-red-50 p-6">
        <div className="flex items-start gap-3">
          <div className="flex h-10 w-10 flex-shrink-0 items-center justify-center rounded-full bg-red-100 text-red-600">
            <svg className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
              />
            </svg>
          </div>
          <div className="flex-1">
            <div className="flex items-center gap-2">
              <h3 className="text-base font-semibold text-red-900">Image Validation Gatekeeper Refusal</h3>
              {validationFailure?.rejection_code && (
                <span className="rounded bg-red-200 px-2 py-0.5 text-xs font-mono font-semibold text-red-800">
                  {validationFailure.rejection_code}
                </span>
              )}
            </div>
            <p className="mt-1 text-sm font-medium text-red-800">{error}</p>
            {validationFailure?.predicted_class && (
              <p className="mt-1 text-xs text-red-700">
                Detected Category: <span className="font-semibold">{validationFailure.predicted_class}</span> (
                {Math.round(validationFailure.confidence * 100)}% confidence)
              </p>
            )}
            <p className="mt-2 text-xs text-red-600">
              Safety Enforcement: Model prediction and Grad-CAM++ generation were suppressed to prevent spurious diagnostics.
            </p>
            {onClear && (
              <button
                type="button"
                onClick={onClear}
                className="mt-4 inline-flex items-center rounded-md bg-red-600 px-3 py-1.5 text-xs font-semibold text-white shadow-sm hover:bg-red-500 transition"
              >
                Upload Valid Pediatric Hand X-Ray
              </button>
            )}
          </div>
        </div>
      </div>
    );
  }

  if (!result) {
    return (
      <div className="card text-center text-slate-500 py-12">
        <p className="text-sm font-medium">Upload a pediatric hand X-ray and click Predict Bone Age</p>
        <p className="mt-1 text-xs text-slate-400">
          All images undergo 3-stage validation (integrity, 6-class anatomy, coverage) prior to model execution.
        </p>
      </div>
    );
  }

  const confidencePct = Math.round(result.confidence * 100);
  const validation = result.validation;

  return (
    <div className="card space-y-4">
      {/* Validation status badge */}
      {validation && (
        <div className="flex items-center justify-between rounded-lg border border-emerald-200 bg-emerald-50 px-3 py-2 text-xs">
          <div className="flex items-center gap-1.5 font-medium text-emerald-800">
            <svg className="h-4 w-4 text-emerald-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <span>Stage 3 Passed: Verified Pediatric Hand X-Ray</span>
          </div>
          <span className="font-semibold text-emerald-700">
            {Math.round(validation.confidence * 100)}% Match
          </span>
        </div>
      )}

      <div>
        <p className="text-sm font-medium text-slate-500">Predicted Bone Age</p>
        <div className="mt-2 flex items-baseline gap-2">
          <span className="text-4xl font-bold text-primary-700">
            {result.bone_age_months}
          </span>
          <span className="text-lg text-slate-500">months</span>
        </div>
        <p className="mt-1 text-sm font-medium text-slate-600">
          ≈ {yearsMonths(result.bone_age_months)}
        </p>
      </div>

      {/* Typical Error (+/- 1 RMSE) Interval */}
      {result.uncertainty_lower_months != null && result.uncertainty_upper_months != null && (
        <div className="rounded-lg border border-sky-200 bg-sky-50/70 p-3">
          <div className="flex items-center justify-between text-xs font-semibold text-sky-900">
            <span>
              {result.uncertainty_note ??
                `typical error about +/-${result.uncertainty_interval_months} months (about 1 RMSE, measured on validation data of unknown split)`}
            </span>
          </div>
          <div className="mt-1 flex items-baseline gap-2 text-sky-800">
            <span className="text-base font-bold">
              [{result.uncertainty_lower_months} – {result.uncertainty_upper_months}]
            </span>
            <span className="text-xs">months</span>
            <span className="text-xs text-sky-600">
              (≈ {yearsMonths(result.uncertainty_lower_months)} – {yearsMonths(result.uncertainty_upper_months)})
            </span>
          </div>
        </div>
      )}

      <div>
        <div className="mb-1 flex justify-between text-xs">
          <span className="font-medium text-slate-600">Model Calibrated Confidence</span>
          <span className="text-slate-500">{confidencePct}%</span>
        </div>
        <div className="h-2 overflow-hidden rounded-full bg-slate-200">
          <div
            className="h-full rounded-full bg-gradient-to-r from-primary-500 to-medical-accent transition-all"
            style={{ width: `${confidencePct}%` }}
          />
        </div>
      </div>

      <div className="flex flex-wrap gap-2 text-xs text-slate-500 pt-2 border-t border-slate-100">
        <span className="rounded-full bg-slate-100 px-2.5 py-1">
          Model: {result.model_type}
        </span>
        {result.inference_engine && (
          <span className="rounded-full bg-primary-50 text-primary-700 border border-primary-200 px-2.5 py-1 font-medium">
            Engine: {result.inference_engine}
          </span>
        )}
        <span className="rounded-full bg-slate-100 px-2.5 py-1">
          Inference: {result.processing_time_ms.toFixed(0)} ms
        </span>
        {validation?.processing_time_ms && (
          <span className="rounded-full bg-slate-100 px-2.5 py-1">
            Validation: {validation.processing_time_ms.toFixed(0)} ms
          </span>
        )}
        {result.gender_used && (
          <span className="rounded-full bg-slate-100 px-2.5 py-1 capitalize">
            Sex: {result.gender_used}
          </span>
        )}
      </div>
    </div>
  );
}
