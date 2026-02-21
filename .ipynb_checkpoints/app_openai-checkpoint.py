import logging
import logger_setup
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from Agent import app

log = logging.getLogger(__name__)

# Pydantic models matching OpenAI format
class Message(BaseModel):
    role: str  # "user", "assistant", "system"
    content: str

class ChatRequest(BaseModel):
    model: str = "Nexus-agent"
    messages: List[Message]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None

class ChatResponse(BaseModel):
    id: str = "Nexus-1"
    object: str = "chat.completion"
    created: int = 0
    model: str = "Nexus-agent"
    choices: list
    usage: dict

# Initialize FastAPI
api = FastAPI(title="Nexus Agent API", version="1.0.0")

# Enable CORS for HuggingFace Chat-UI
api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

config = {"configurable": {"thread_id": "1"}}

@api.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    try:
        # Extract last user message
        user_message = request.messages[-1].content if request.messages else ""
        log.info(f"Received message: ```{user_message}``` and the request object: {request}")
        if not user_message:
            raise HTTPException(status_code=400, detail="No user message provided")
        
        # Call your agent
        response = app.invoke(
            {"messages": [{"role": "user", "content": user_message}]},
            config=config
        )

        log.info(f"Agent response object: {response}")
                
        #print the conversation in a readable format
        for message in response["messages"]:
                message.pretty_print()


        # Robustly extract the final assistant text from the agent response.
        def extract_final_text(resp):
            msgs = resp.get('messages', []) if isinstance(resp, dict) else getattr(resp, 'messages', [])
            # iterate from the end and skip messages that look like tool calls
            for m in reversed(msgs):
                # handle dict-like messages
                if isinstance(m, dict):
                    # skip tool invocation entries that typically contain a `name` key
                    if m.get('name'):
                        continue
                    content = m.get('content') or m.get('message') or m.get('text')
                else:
                    # object with attributes
                    if hasattr(m, 'name') and getattr(m, 'name'):
                        continue
                    content = getattr(m, 'content', None) or getattr(m, 'text', None)

                if not content:
                    continue
                text = content if isinstance(content, str) else str(content)
                stripped = text.strip()
                # skip JSON-like or tool-like payloads
                if stripped.startswith('{') or stripped.startswith('[') or stripped.startswith('"'):
                    continue
                return text

            # fallback to last message if nothing else
            if msgs:
                m = msgs[-1]
                if isinstance(m, dict):
                    return m.get('content') or str(m)
                return getattr(m, 'content', None) or str(m)
            return ""
        # agent_response = extract_final_text(response)
        agent_response = getattr(response["messages"][-1] , "content")

        print(f"Returning:- \n{agent_response}")
        
        return {
            "id": "Nexus-1",
            "object": "chat.completion",
            "created": 0,
            "model": "Nexus-agent",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": agent_response},
                    "finish_reason": "stop"
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        }
    except Exception as e:
        log.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@api.get("/v1/models")
async def list_models():
    return {"data": [{"id": "Nexus-agent", "object": "model"}]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(api, host="0.0.0.0", port=8000)