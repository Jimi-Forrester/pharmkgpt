from fastapi import FastAPI, APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field # Use Field for better validation/docs if needed
import threading
import json
import logging
import uvicorn # For running the app


class MockRAGEngine:
    def __init__(self):
        self._params = {}
        print("MockRAGEngine Initialized")

    def setup_query_engine(self, model_type=None, api_key=None, top_k=None, hops=None):
        print(f"MockRAGEngine: Setting up with model_type={model_type}, api_key=***, top_k={top_k}, hops={hops}")
        self._params = {
            "model_type": model_type,
            "api_key_present": bool(api_key), # Don't store actual key
            "top_k": top_k,
            "hops": hops
        }
        # Simulate some setup work
        import time
        time.sleep(0.1)
        print("MockRAGEngine: Setup complete.")

    def query(self, question: str):
        print(f"MockRAGEngine: Querying with question: {question}")
        # Simulate streaming results
        yield {"type": "retrieval", "docs": ["doc1", "doc2"]}
        import time
        time.sleep(0.5)
        yield {"type": "generation", "chunk": "This is the first part..."}
        time.sleep(0.5)
        yield {"type": "generation", "chunk": " and this is the second."}
        yield {"type": "final_answer", "answer": "This is the first part... and this is the second."}
        yield {"type": "sources", "sources": ["doc1", "doc2"]}


    def get_parameters(self):
        print("MockRAGEngine: Getting parameters.")
        return self._params

# Replace MockRAGEngine with your actual RAGEngine import
RAGEngine = MockRAGEngine

# --- Pydantic Models for Request Bodies ---
class QueryRequest(BaseModel):
    question: str

class ConfigUpdateRequest(BaseModel):
    # Use Optional[] or | None for optional fields
    # Add default values if applicable
    model_type: str | None = None
    api_key: str | None = None
    top_k: int | None = None
    hops: int | None = None

# --- FastAPI Application Setup ---
app = FastAPI(title="RAG Engine API", description="API for interacting with the RAG Engine")


app.state.rag_engine = None
app.state.initialization_lock = threading.Lock()

# --- Logging Setup ---
# FastAPI uses standard Python logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Dependency Function to Get Engine ---
# This function will be used by routes that need the engine.
# It ensures the engine is initialized before the route logic runs.
async def get_initialized_engine(request: Request) -> RAGEngine:
    engine = request.app.state.rag_engine
    if engine is None:
        raise HTTPException(
            status_code=503, # Service Unavailable
            detail="RAG Engine not initialized. Please call /api/update_config first."
        )
    return engine

# --- API Router (Equivalent to Flask Blueprint) ---
# Using a router helps organize endpoints, similar to Blueprint
router = APIRouter(prefix="/api") # Set prefix for all routes in this router

# --- Route Implementations ---

@router.post("/query")
async def query_rag_endpoint( # Use async def for FastAPI routes, especially I/O bound
    payload: QueryRequest, # FastAPI injects validated data from request body
    rag_engine: RAGEngine = Depends(get_initialized_engine) # Use dependency injection
):
    """
    Receives a question and streams back the RAG engine's response.
    """
    question = payload.question

    async def generate_response_stream(): # Must be async if rag_engine.query is async
                                          # If rag_engine.query is sync, keep this sync
                                          # but the route MUST be async for StreamingResponse
        try:
            # Check if rag_engine.query is an async generator or sync generator
            # Assuming it's a synchronous generator here based on Flask code
            # If it were async: async for item in rag_engine.query(question):
            result_generator = rag_engine.query(question)
            for item in result_generator:
                json_data = json.dumps(item)
                # Format for Server-Sent Events (SSE) if that's intended
                # The original Flask mimetype='text/event-stream' suggests SSE
                yield f"data: {json_data}\n\n" # SSE format: "data: <json>\n\n"

        except Exception as e:
            logger.error(f"Error during query stream generation: {e}", exc_info=True)
            # Send an error event via SSE
            error_message = {"type": "error", "message": f"An error occurred: {str(e)}"}
            yield f"data: {json.dumps(error_message)}\n\n"

    # Use FastAPI's StreamingResponse
    # media_type='text/event-stream' is crucial for SSE clients
    return StreamingResponse(generate_response_stream(), media_type='text/event-stream')


@router.post('/update_config')
async def update_engine_config_endpoint(
    payload: ConfigUpdateRequest,
    request: Request # Need Request object to access app.state
):
    """
    Initializes or updates the RAG Engine configuration.
    This operation is thread-safe.
    """
    # Access lock and engine instance via app.state attached to the request
    lock = request.app.state.initialization_lock
    rag_engine_instance = request.app.state.rag_engine # Get current instance state

    try:
        # Acquire lock to ensure atomic initialization/update
        with lock:
            # Re-check engine inside lock in case it was initialized by another thread
            rag_engine_instance = request.app.state.rag_engine

            if rag_engine_instance is None:
                logger.info("Attempting to initialize RAGEngine for the first time...")
                new_engine = RAGEngine()
                new_engine.setup_query_engine(
                    model_type=payload.model_type,
                    api_key=payload.api_key,
                    top_k=payload.top_k,
                    hops=payload.hops
                )
                # Store the successfully initialized engine on the app state
                request.app.state.rag_engine = new_engine
                logger.info("Engine initialized successfully.")
                message = "Engine initialized successfully"
            else:
                logger.info("Attempting to update existing RAGEngine configuration...")
                # Call update method on the existing engine instance
                rag_engine_instance.setup_query_engine(
                    model_type=payload.model_type,
                    api_key=payload.api_key,
                    top_k=payload.top_k,
                    hops=payload.hops
                )
                logger.info("RAGEngine configuration updated.")
                message = "Configuration updated successfully"

        # Fetch the potentially new/updated engine instance AFTER the lock is released
        final_engine = request.app.state.rag_engine
        if final_engine is None: # Should not happen if logic above is correct
            raise HTTPException(status_code=500, detail="Engine instance is unexpectedly None after configuration attempt.")

        # Return success response outside the lock
        return JSONResponse( # Use JSONResponse for explicit status code and content
            status_code=200,
            content={
                "message": message,
                # Get parameters from the engine instance now stored in app.state
                "new_params": final_engine.get_parameters()
            }
        )

    except Exception as e:
        logger.error(f"Error during engine initialization/update: {e}", exc_info=True)
        # Use HTTPException for standard FastAPI error responses
        raise HTTPException(
            status_code=500,
            detail=f"Failed to initialize/update configuration: {str(e)}"
        )


@router.get('/current_params')
async def get_current_params_endpoint(
    rag_engine: RAGEngine = Depends(get_initialized_engine) # Use dependency injection
):
    """
    Returns the current parameters of the initialized RAG Engine.
    """
    # The Depends(get_initialized_engine) ensures engine exists
    # and raises HTTPException 503 otherwise.
    params = rag_engine.get_parameters()
    return JSONResponse(content=params) # Return parameters as JSON


# --- Include the router in the main FastAPI app ---
app.include_router(router)

# --- Add a root endpoint for basic check ---
@app.get("/")
async def read_root():
    return {"message": "Welcome to the RAG Engine API"}

# --- Run the app with Uvicorn ---
# This block allows running directly with `python main.py`
# For production, use `uvicorn main:app --host 0.0.0.0 --port 8000`
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)