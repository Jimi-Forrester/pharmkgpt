from flask import Blueprint, request, jsonify, current_app
# Assuming RAGEngine class is imported if needed for type hinting or direct use
from .rag import RAGEngine 

bp = Blueprint("api", __name__)

@bp.route("/query", methods=["POST"])
def query_rag():
    # --- Key Change 4: Check if engine is initialized ---
    rag_engine = current_app.rag_engine
    if rag_engine is None:
        return jsonify({"error": "RAG Engine not initialized. Please call /api/update_config with initial parameters first."}), 503 # 503 Service Unavailable

    data = request.get_json()
    if not data or "question" not in data:
        return jsonify({"error": "Missing question"}), 400
    question = data["question"]

    try:
        result = rag_engine.query(question)
        return jsonify(result)
    except Exception as e:
        current_app.logger.error(f"Error during query: {e}", exc_info=True) # Log the full error
        return jsonify({"error": str(e)}), 500


@bp.route('/update_config', methods=['POST'])
def update_engine_config():
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid JSON payload"}), 400

    # Extract parameters (add validation as needed)
    model_type = data.get('model_type')
    api_key = data.get('api_key')
    top_k = data.get('top_k')
    hops = data.get('hops')
    
    # --- Key Change 3: Thread-safe Initialization/Update ---
    try:
        # Acquire lock to ensure only one thread initializes/updates at a time
        with current_app.initialization_lock:
            if current_app.rag_engine is None:
                # --- First time initialization ---
                print("Attempting to initialize RAGEngine for the first time...")
                # Create the instance
                new_engine = RAGEngine() 
                # Call its initialize method
                new_engine.initialize(
                    model_type=model_type,
                    api_key=api_key,
                    top_k=top_k,
                    hops=hops
                )
                # Store the successfully initialized engine
                current_app.rag_engine = new_engine 
                print("RAGEngine initialized successfully.")
                message = "Engine initialized successfully"
            else:
                # --- Update existing engine ---
                print("Attempting to update existing RAGEngine configuration...")
                current_app.rag_engine.initialize( # Or use an update method if available
                    model_type=model_type,
                    api_key=api_key,
                    top_k=top_k,
                    hops=hops
                )
                print("RAGEngine configuration updated.")
                message = "Configuration updated successfully"

        # Return success response outside the lock
        return jsonify({
            "message": message, 
            "new_params": current_app.rag_engine.get_parameters() # Get params from the now-guaranteed-to-exist engine
        }), 200

    except Exception as e:
        current_app.logger.error(f"Error during engine initialization/update: {e}", exc_info=True)
        return jsonify({"error": f"Failed to initialize/update configuration: {str(e)}"}), 500


@bp.route('/current_params')
def get_current_params_route():
    # Check if initialized first
    rag_engine = current_app.rag_engine
    if rag_engine is None:
        return jsonify({"error": "RAG Engine not initialized."}), 503
    return jsonify(rag_engine.get_parameters())