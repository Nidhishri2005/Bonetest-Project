"""Step 4 App Smoke Test: scripts/smoke_test_api.py

Starts the FastAPI backend (uvicorn app.main:app) on 127.0.0.1:8011, waits for
/api/health to return 200 OK, and sends real HTTP POST /api/predict requests for:
  1. 5 real RSNA images NOT in any excluded list (neither in
     configs/validator_excluded_eval_files.json nor in the 4,138 fit set nor in
     the 500 held-out calibration set).
  2. 5 images from tests/validator/should_reject (mix of non-hand X-rays,
     animal photos, adult-hand photos, and screenshots).

Prints a table with HTTP status, error/reason code, deciding stage, OOD kNN score,
threshold, user message, and predicted bone age (months), then stops the server.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import List

import httpx

ROOT_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = ROOT_DIR / "backend"
for p in (str(ROOT_DIR), str(BACKEND_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from ml.validator_ood import (
    build_stratified_fit_and_holdout,
    load_full_rsna_dataframe,
    resolve_rsna_paths,
)


def select_5_unseen_rsna_images() -> List[Path]:
    rsna_dir, csv_path = resolve_rsna_paths()
    full_df = load_full_rsna_dataframe(csv_path)

    excl_path = ROOT_DIR / "configs" / "validator_excluded_eval_files.json"
    with open(excl_path, "r", encoding="utf-8") as f:
        excl_data = json.load(f)

    excluded = set()
    excluded.update(excl_data.get("should_accept_files", []))
    excluded.update(excl_data.get("check_200_files", []))
    excluded.update(excl_data.get("should_reject_files", []))

    fit_df, holdout_df = build_stratified_fit_and_holdout(
        df=full_df,
        rsna_dir=rsna_dir,
        excluded_files=excluded,
        fit_per_bin=1000,
        holdout_total=500,
        infant_holdout=25,
        seed=42,
    )
    used_in_fit_or_holdout = set(fit_df["file_name"].tolist()) | set(holdout_df["file_name"].tolist())
    all_excluded = excluded | used_in_fit_or_holdout

    unseen_df = full_df[~full_df["file_name"].isin(all_excluded)].copy()
    unseen_df = unseen_df[unseen_df["file_name"].apply(lambda fn: (rsna_dir / fn).exists())].reset_index(drop=True)

    # Pick 5 deterministic unseen RSNA images across varied ages
    chosen_rows = unseen_df.iloc[:: max(1, len(unseen_df) // 5)][:5]
    paths = [rsna_dir / fn for fn in chosen_rows["file_name"].tolist()]
    for _, r in chosen_rows.iterrows():
        assert r["file_name"] not in excluded, f"{r['file_name']} in excluded json!"
        assert r["file_name"] not in used_in_fit_or_holdout, f"{r['file_name']} in fit/holdout!"
    return paths


def main() -> None:
    unseen_rsna = select_5_unseen_rsna_images()
    reject_dir = ROOT_DIR / "tests" / "validator" / "should_reject"
    # Select 5 images from should_reject covering Stage 2 OOD rejections and Stage 1 color rejections
    reject_names = ["4.jpe", "5.jpe", "7.jpe", "10.jpe", "22.jpe"]
    reject_paths = [reject_dir / fn for fn in reject_names]

    port = 8011
    base_url = f"http://127.0.0.1:{port}"
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "app.main:app",
        "--app-dir",
        str(BACKEND_DIR),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]

    print(f"Starting FastAPI backend: {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    try:
        client = httpx.Client(timeout=30.0)
        ready = False
        for _ in range(40):
            try:
                r = client.get(f"{base_url}/api/health")
                if r.status_code == 200:
                    ready = True
                    print(f"Backend ready at {base_url}/api/health -> {r.status_code} {r.json()}")
                    break
            except Exception:
                pass
            time.sleep(0.5)

        if not ready:
            out, _ = proc.communicate(timeout=5)
            raise RuntimeError(f"Backend failed to start within 20s. Output:\n{out}")

        print("\n" + "=" * 124)
        print("STEP 4 SMOKE TEST: POST /api/predict (5 Unseen RSNA Images + 5 should_reject Images)")
        print("=" * 124)
        print(
            f"{'Group':<16} | {'File':<12} | {'HTTP':<5} | {'Reason/Error':<14} | "
            f"{'Stage':<16} | {'kNN Score':<10} | {'Threshold':<10} | {'Bone Age (mo) / Message'}"
        )
        print("-" * 124)

        for group_label, paths in [
            ("RSNA (unseen)", unseen_rsna),
            ("should_reject", reject_paths),
        ]:
            for p in paths:
                with open(p, "rb") as f:
                    files = {"file": (p.name, f.read(), "image/jpeg" if p.suffix.lower() == ".jpe" else "image/png")}
                data = {"model_type": "efficientnet_b3", "gender": "male"}
                resp = client.post(f"{base_url}/api/predict", files=files, data=data)
                body = resp.json()

                if resp.status_code == 200:
                    val = body.get("validation") or {}
                    gd = val.get("gate_details") or {}
                    score = val.get("score") if val.get("score") is not None else gd.get("score")
                    thr = val.get("threshold")
                    stage = val.get("deciding_stage") or gd.get("deciding_stage", "passed")
                    bone_age = body.get("bone_age_months")
                    ci_low = body.get("uncertainty_lower_months")
                    ci_high = body.get("uncertainty_upper_months")
                    outcome = f"bone_age={bone_age} mo (90% CI: [{ci_low}, {ci_high}])"
                    err_code = "NONE (OK)"
                else:
                    err_code = body.get("reason_code") or body.get("error") or "ERROR"
                    score = body.get("score")
                    thr = body.get("threshold")
                    stage = body.get("stage")
                    outcome = f"msg=\"{body.get('message')}\""

                score_s = f"{score:.6f}" if isinstance(score, (int, float)) else str(score)
                thr_s = f"{thr:.6f}" if isinstance(thr, (int, float)) else str(thr)
                print(
                    f"{group_label:<16} | {p.name:<12} | {resp.status_code:<5d} | {err_code:<14} | "
                    f"{str(stage):<16} | {score_s:<10} | {thr_s:<10} | {outcome}"
                )
        print("=" * 124)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    main()
