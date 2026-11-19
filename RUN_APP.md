# Running BoneTest Locally on Windows (Backend + Frontend)

This guide explains how to start both the **FastAPI Backend** (`port 8000`) and the **React + Vite Frontend** (`port 5173`) on this Windows PC so you can access the full application in your browser.

---

## 1. Two Commands to Run (in Two VS Code Terminals)

Open two terminals in VS Code (`Ctrl+Shift+\`` or **Terminal → New Terminal**), both starting in `c:\bonetest`:

### Terminal 1 — Start the FastAPI Backend (`port 8000`)

**Using the batch script (Command Prompt or PowerShell):**
```powershell
.\start_backend.bat
```

*(Or run the command directly in PowerShell:)*
```powershell
.\.venv\Scripts\Activate.ps1
python -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000
```

Wait until you see:
```text
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

---

### Terminal 2 — Start the React/Vite Frontend (`port 5173`)

**Using the batch script (Command Prompt or PowerShell):**
```powershell
.\start_frontend.bat
```

*(Or run the commands directly in PowerShell — note that Node.js is installed at `C:\Users\Manya\nodejs\node-v22.13.1-win-x64`, so add it to `PATH` first:)*
```powershell
$env:PATH = "C:\Users\Manya\nodejs\node-v22.13.1-win-x64;$env:PATH"
cd c:\bonetest\frontend
npm.cmd run dev
```

Wait until you see:
```text
  VITE v6.4.3  ready in ... ms

  ➜  Local:   http://localhost:5173/
```

---

## 2. URLs to Open in Your Browser

| Service | URL | Purpose |
| :--- | :--- | :--- |
| **Frontend Web App** | **http://localhost:5173** | Main BoneTest UI (upload X-rays, predict bone age, view Grad-CAM++ & metrics) |
| **Backend Swagger Docs** | **http://localhost:8000/docs** | Interactive FastAPI OpenAPI documentation |
| **Backend Health Check** | **http://localhost:8000/api/health** | JSON health check (`{"status":"ok","version":"1.0.0","models_loaded":[]}`) |

---

## 3. Three Most Likely Errors & Fixes

### Error 1: `'npm' is not recognized as the name of a cmdlet, function, script file, or operable program`
- **Cause:** On this PC, Node.js (`v22.13.1`) is installed in `C:\Users\Manya\nodejs\node-v22.13.1-win-x64`, which is not in the default system `PATH`.
- **Fix:** Run `.\start_frontend.bat` from `c:\bonetest` (which automatically prepends `C:\Users\Manya\nodejs\node-v22.13.1-win-x64` to `PATH`), or run this in PowerShell before `npm.cmd run dev`:
  ```powershell
  $env:PATH = "C:\Users\Manya\nodejs\node-v22.13.1-win-x64;$env:PATH"
  ```

### Error 2: `ERROR: [Errno 10048] error while attempting to bind on address ('127.0.0.1', 8000): only one usage of each socket address is normally permitted` (or Vite moves from port `5173` to `5174`)
- **Cause:** An older backend (`port 8000`) or frontend (`port 5173`) process is already running in another terminal or background process. Note that backend CORS (`CORS_ORIGINS=http://localhost:5173,http://127.0.0.1:5173`) is configured for port `5173`.
- **Fix:** Stop the existing process occupying port `8000` or `5173` in PowerShell:
  ```powershell
  Get-NetTCPConnection -LocalPort 8000,5173 -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }
  ```
  Then restart `.\start_backend.bat` and `.\start_frontend.bat`.

### Error 3: Frontend shows `"Unable to connect to API"` or `ECONNREFUSED http://localhost:8000`
- **Cause:** The frontend (`http://localhost:5173`) proxies `/api/*` requests to `http://localhost:8000` (configured in `frontend/vite.config.ts`). If the backend server in Terminal 1 is not running on port `8000` (or failed startup because `backend/checkpoints/validator_ood.npz` was missing), `/api/models` and `/api/predict` will fail.
- **Fix:**
  1. Ensure Terminal 1 (`.\start_backend.bat`) is running and `http://localhost:8000/api/health` returns `{"status":"ok",...}`.
  2. Verify `c:\bonetest\backend\checkpoints\validator_ood.npz` exists (`8,672,200` bytes).
