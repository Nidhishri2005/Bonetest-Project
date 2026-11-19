import type { ModelInfo, ModelType } from "../types";

interface ModelSelectorProps {
  models: ModelInfo[];
  selected: ModelType;
  onChange: (model: ModelType) => void;
  gender: string;
  onGenderChange: (gender: string) => void;
  disabled?: boolean;
}

export default function ModelSelector({
  models,
  selected,
  onChange,
  gender,
  onGenderChange,
  disabled,
}: ModelSelectorProps) {
  const current = models.find((m) => m.id === selected);

  return (
    <div className="space-y-4">
      <div>
        <label className="mb-1.5 block text-sm font-medium text-slate-700">
          Model Architecture
        </label>
        <select
          value={selected}
          onChange={(e) => onChange(e.target.value as ModelType)}
          disabled={disabled}
          className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2.5 text-sm
                     focus:border-primary-500 focus:outline-none focus:ring-2 focus:ring-primary-200"
        >
          {models.map((m) => (
            <option
              key={m.id}
              value={m.id}
              disabled={!m.checkpoint_available}
            >
              {m.is_legacy ? `[Legacy] ${m.display_name}` : `★ ${m.display_name}`}
              {!m.checkpoint_available ? " (unavailable - no checkpoint)" : ""}
            </option>
          ))}
        </select>
        {current && (
          <p className="mt-1.5 text-xs text-slate-500">{current.description}</p>
        )}
      </div>

      {current?.requires_gender && (
        <div className={`rounded-lg border p-3.5 ${current.id === "efficientnet_b3" ? "border-primary-200 bg-primary-50/50" : "border-amber-200 bg-amber-50"}`}>
          <div className="mb-1.5 flex items-center justify-between">
            <label className={`block text-xs font-semibold uppercase tracking-wide ${current.id === "efficientnet_b3" ? "text-primary-900" : "text-amber-900"}`}>
              Biological Sex Conditioning
            </label>
            <span className={`rounded px-1.5 py-0.5 text-[10px] font-semibold ${current.id === "efficientnet_b3" ? "bg-primary-100 text-primary-800" : "bg-amber-200 text-amber-900"}`}>
              {current.id === "efficientnet_b3" ? "Required for Target Model" : "Experimental"}
            </span>
          </div>
          <p className={`mb-3 text-xs ${current.id === "efficientnet_b3" ? "text-primary-700" : "text-amber-700"}`}>
            {current.id === "efficientnet_b3"
              ? "Sex conditioning provides clinical maturation curves (+1.33 months MAE improvement over image-only)."
              : "Standard legacy models are image-only. This experimental model tests metadata fusion with biological sex."}
          </p>
          <div className="flex gap-3">
            {(["male", "female"] as const).map((g) => (
              <label
                key={g}
                className={`flex flex-1 cursor-pointer items-center justify-center rounded-lg border px-4 py-2 text-sm font-medium capitalize transition
                  ${gender === g ? "border-primary-500 bg-white text-primary-700 shadow-sm" : "border-slate-300 bg-slate-100 text-slate-600 hover:bg-white"}`}
              >
                <input
                  type="radio"
                  name="gender"
                  value={g}
                  checked={gender === g}
                  onChange={() => onGenderChange(g)}
                  className="sr-only"
                  disabled={disabled}
                />
                {g}
              </label>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
