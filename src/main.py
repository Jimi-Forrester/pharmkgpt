import asyncio
import threading
import json
import logging
from contextlib import asynccontextmanager

from .rag import RAGEngine
from fastapi import FastAPI, APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# --- Pydantic 模型 (保持不变) ---
import asyncio
import threading
import json
import logging
from typing import Optional
from contextlib import asynccontextmanager

# FastAPI Imports
from fastapi import FastAPI, APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
import uvicorn

from google.api_core import exceptions
from langchain_community.llms.ollama import OllamaEndpointNotFoundError

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# --- Pydantic Models (Keep as before) ---
class QueryRequest(BaseModel):
    question: str

class ConfigUpdateRequest(BaseModel):
    model_type: str | None = Field(default=None, description="Model type (e.g., 'gemini', 'gemma2', 'mistral'). See config.py for options.")
    api_key: str | None = Field(default=None, description="API key (required only for certain models like Gemini).")
    top_k: int | None = Field(default=5, description="Number of documents to retrieve initially.")
    hops: int | None = Field(default=1, description="Number of hops for KG traversal.")

class EngineParams(BaseModel):
    model_type: Optional[str] = None
    api_key_provided: Optional[bool] = None
    top_k: Optional[int] = None
    hops: Optional[int] = None
    

class EngineStatusResponse(BaseModel):
    """定义 /api/status 响应的结构"""
    configured: bool
    params: Optional[EngineParams] = None # 参数部分是可选的
    
    
# --- Lifespan Event Manager (Adjusted for actual RAGEngine) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application startup: Initializing RAGEngine instance and config lock...")
    rag_engine_instance = None
    try:
        # Instantiate the engine - __init__ will load KG/Index here
        rag_engine_instance = RAGEngine()
        app.state.rag_engine = rag_engine_instance
        app.state.config_lock = threading.Lock()
        logger.info("Application startup: RAGEngine instance created and lock initialized.")
        logger.info("Engine is ready for configuration via /api/update_config.")
        
    except (RuntimeError, FileNotFoundError) as e:
        logger.critical(f"Application startup FAILED: Could not initialize RAGEngine during startup: {e}", exc_info=True)
        # Optional: Exit or prevent app from fully starting if engine init is critical
        app.state.rag_engine = None # Ensure state is None if init fails
        app.state.config_lock = None
        # Raising an exception here might stop FastAPI startup cleanly
        raise RuntimeError(f"Critical RAGEngine initialization failed: {e}") from e
    except Exception as e:
        logger.critical(f"Application startup FAILED: Unexpected error during RAGEngine initialization: {e}", exc_info=True)
        app.state.rag_engine = None
        app.state.config_lock = None
        raise RuntimeError(f"Unexpected critical error during startup: {e}") from e

    yield # Application runs here

    logger.info("Application shutdown: Cleaning up...")
    engine = getattr(app.state, 'rag_engine', None)
    if engine and hasattr(engine, 'close'): # Check if you added a close method
        try:
            logger.info("Closing RAGEngine resources...")
            # If close becomes async: await engine.close()
            engine.close()
            logger.info("RAGEngine resources closed.")
        except Exception as e:
            logger.error(f"Error during RAGEngine cleanup: {e}", exc_info=True)
    app.state.rag_engine = None
    app.state.config_lock = None
    logger.info("Application shutdown complete.")


# --- FastAPI App Setup ---
app = FastAPI(
    title="RAG Engine API",
    description="API for interacting with the RAG Engine, including KG features.",
    lifespan=lifespan
)

# --- Dependency Functions (Adjusted for actual RAGEngine) ---
async def get_engine_instance(request: Request) -> RAGEngine:
    """Gets the RAGEngine instance from app state (doesn't check config)."""
    engine = getattr(request.app.state, 'rag_engine', None)
    if not engine:
        # This should ideally not happen if lifespan succeeded
        logger.error("Critical error: RAGEngine instance not found in app state.")
        raise HTTPException(status_code=500, detail="Internal Server Error: RAG engine instance is not available.")
    return engine

async def get_config_lock(request: Request) -> threading.Lock:
    """Gets the configuration lock from app state."""
    lock = getattr(request.app.state, 'config_lock', None)
    if not lock:
        logger.error("Critical error: Configuration lock not found in app state.")
        raise HTTPException(status_code=500, detail="Internal Server Error: Configuration lock is not available.")
    return lock

async def get_configured_engine(
    engine: RAGEngine = Depends(get_engine_instance)
) -> RAGEngine:
    """Gets the RAGEngine instance and verifies it's configured."""
    return engine

# --- API Router ---
router = APIRouter(prefix="/api")

# --- Route Implementations (Adapted for actual RAGEngine) ---

@router.post("/query")
async def query_rag_endpoint(
    payload: QueryRequest,
    rag_engine: RAGEngine = Depends(get_configured_engine) # Ensures engine is configured
):

    question = payload.question
    logger.info(f"Received API query request for: '{question}'")

    async def generate_response_stream():
        # try:
            
        for item in rag_engine.query(question):
            try:
                json_data = json.dumps(item)
                yield json_data + "\n"
                
            except TypeError as json_err:
                # Handle cases where item might not be JSON serializable
                logger.error(f"Failed to serialize item to JSON: {item} - Error: {json_err}")
                error_msg = {"type": "error", "message": f"Internal error: Could not serialize response chunk. {json_err}"}
                yield json.dumps(error_msg)

        # except HTTPException:
        #     raise
        
        # except Exception as e:
        #     logger.error(f"Error during query stream generation for '{question}': {e}", exc_info=True)
        #     error_message = {"type": "error", "message": f"An error occurred during streaming: {str(e)}"}
        #     try:
        #         yield json.dumps(error_message)
        #     except Exception as final_e:
        #         logger.error(f"CRITICAL: Could not even send error message via SSE: {final_e}")
    return StreamingResponse(generate_response_stream(), media_type='text/event-stream')


@router.post('/update_config')
async def update_engine_config_endpoint(
    payload: ConfigUpdateRequest,
    rag_engine: RAGEngine = Depends(get_engine_instance), # Get instance
    lock: threading.Lock = Depends(get_config_lock)       # Get lock
):
    """
    Configures or updates the RAG Engine settings (LLM, top_k, hops). Thread-safe.
    """
    logger.info(f"Received config update request: {payload.model_dump(exclude_unset=True)}")

    if lock.locked():
        logger.warning("Configuration update endpoint is busy. Client may need to retry.")
        raise HTTPException(status_code=503, detail="Configuration update is already in progress. Please try again shortly.")
    
    try:
        with lock:
            logger.info("Configuration lock acquired. Updating RAGEngine setup...")
            try:
                # Call the actual setup method on the existing instance
                rag_engine.setup_query_engine(
                    model_type=payload.model_type,
                    api_key=payload.api_key,
                    top_k=payload.top_k,
                    hops=payload.hops
                )
                logger.info("RAGEngine setup/update successful.")
                message = "Engine configuration updated successfully."
                status_code = 200

            except (ValueError, FileNotFoundError) as config_err:
                logger.error(f"Configuration failed: {config_err}", exc_info=True)
                raise HTTPException(status_code=500, detail=f"Configuration Error: {str(config_err)}") from config_err
            except RuntimeError as setup_err: # Catch the generic setup error
                logger.error(f"Runtime error during engine setup: {setup_err}", exc_info=True)
                raise HTTPException(status_code=500, detail=f"Engine Setup Failed: {str(setup_err)}") from setup_err
            except exceptions.GoogleAPIError as e:
                raise HTTPException(
                    status_code=500, 
                    detail = f"API key not valid. Please pass a valid API key. ")
            except Exception as e:
                logger.exception("Unexpected error during engine setup within lock.") # Use logger.exception to include traceback
                raise HTTPException(status_code=500, detail=f"{str(e)}") from e

        # Return success response outside the lock
        return JSONResponse(
            status_code=status_code,
            content={
                "message": message,
                "new_params": rag_engine.get_status()['params'] # Get updated params
            }
        )
    except HTTPException:
        # Allow HTTPExceptions raised within the 'with lock' block or from dependencies to propagate
        raise
    except Exception as e:
        # Catch errors acquiring the lock or other unexpected issues outside the core setup
        logger.error(f"Error in update_config endpoint outside lock/setup: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An unexpected error occurred processing the configuration request.")


@app.get(
    "/api/status",
    response_model=EngineStatusResponse, # 指定响应模型
    summary="Get RAG Engine Status",
    description="Retrieves the current configuration status and parameters of the RAG Engine instance."
)
async def get_engine_status(
    rag_engine: RAGEngine = Depends(get_engine_instance), 
):
    """
    获取 RAGEngine 的当前状态。
    前端可以在页面加载或刷新时调用此接口。
    """
    status_data = rag_engine.get_status()
    print(f"Sending status to frontend: {status_data}")
    # FastAPI 会自动将返回的字典与 Pydantic 模型匹配并转换为 JSON
    return JSONResponse(status_data) 


# --- Include Router ---
app.include_router(router)



# --- Main Execution ---
if __name__ == "__main__":
    print("Starting Uvicorn server...")
    uvicorn.run("main:app", host="127.0.0.1", port=5000, reload=True)