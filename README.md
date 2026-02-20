# Nexus Agent — Local OpenAI-compatible API

Small project that exposes an agent (GLM + KNN models + LangGraph/ LangChain agent) behind an OpenAI-compatible REST API so it can be connected to UIs such as HuggingFace Chat-UI or OpenWebUI.

Quick notes
- API server: `app_openai.py` — provides `POST /v1/chat/completions` and `GET /v1/models`.
- Frontend: OpenWebUI (optional) — can proxy to the local API via its OpenAI-compatible connection.
- For simplicity in this assignment the Hugging Face token (`HF_TOKEN`) is set directly in `Agent.py` (not production-safe).

### Quick start (backend)
1. Create and activate a Python 3.11 virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Add your Hugging Face token to `Agent.py` (replace the placeholder with your actual `HF_TOKEN`).

4. Start the API server:Start the API server:

```bash
python app_openai.py
```

5. The service exposes an OpenAI-compatible endpoint at `/v1/chat/completions` (default port in `app_openai.py` is `8000` in code).


### Connecting a UI
- Configure your UI (OpenWebUI / HuggingFace Chat-UI) to use an OpenAI API URL pointing at your running backend (example: `http://localhost:8000/` or `http://localhost:8080/` depending on how you run/forward ports).

Files of interest
- `Agent.py` — agent, tools, and model wiring.
- `app_openai.py` — OpenAI-compatible REST wrapper.
- `GLMModel.py`, `KNN.py` — model logic.
- `logger_setup.py` — basic stdout logger configuration.

Security note
- This repository stores our `HF_TOKEN` directly in `Agent.py` (which will expire after commit) for simplicity. 
For any real deployment, use environment variables or a `secrets manager` and never commit secrets to source control.
