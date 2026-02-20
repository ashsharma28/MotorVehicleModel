# Deployment notes — Frontend (OpenWebUI) and Backend (API)

This document describes how to start the frontend (OpenWebUI) and the backend (OpenAI-compatible API server) locally for this assignment.

Frontend — OpenWebUI

1. Create a Python 3.11 virtual environment (recommended):

```bash
python -m venv openwebui-env
```

2. Activate the environment:

Windows (PowerShell):

```powershell
.\openwebui-env\Scripts\Activate.ps1
```

3. Install OpenWebUI:

```bash
pip install open-webui
```

4. Start the OpenWebUI server:

```bash
open-webui serve
```

5. Configure OpenWebUI to use this project's backend as an OpenAI-compatible connection:
- Open the OpenWebUI settings → Connections → Add New → OpenAI API
- For the API URL use: `http://localhost:8080/` (or `http://localhost:8000/` if your backend runs on port 8000). Pointing to the running backend allows the UI to send chat requests to the local model API.


Backend — OpenAI-compatible API

1. Create a separate Python virtual environment for the backend:

```bash
python -m venv backend-env
```

2. Activate it:

Windows (PowerShell):

```powershell
.\backend-env\Scripts\Activate.ps1
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Start the API server (the repo includes `app_openai.py`):

```bash
python app_openai.py
```

By default the server runs with `uvicorn` inside `app_openai.py`. Adjust the host/port there if you need the API to run on a different port.

Notes and caveats
- For simplicity in this assignment the Hugging Face token (`HF_TOKEN`) is set directly in `Agent.py`. This is NOT secure for production. Use environment variables or a secrets manager for real deployments.
- If OpenWebUI or other UIs appear to return intermediate tool-call messages instead of the final assistant text, ensure the backend extracts the final assistant reply (see `app_openai.py` which includes logic to skip tool messages).
- Ensure CORS is enabled (already added to `app_openai.py`) when accessing the backend from a browser-hosted UI.
