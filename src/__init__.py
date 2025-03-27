import threading
from flask import Flask, jsonify
from .routes import bp as api_bp
from .rag import RAGEngine # Assuming your engine class is here

def create_app():
    app = Flask(__name__)

    # --- Key Change 1: Initialize engine to None ---
    app.rag_engine = RAGEngine() 
    # --- Key Change 2: Add a lock for thread-safe initialization ---
    app.initialization_lock = threading.Lock() 
    print("RAGEngine initialized successfully.")

    # Register routes
    app.register_blueprint(api_bp, url_prefix='/api')

    # Add a status endpoint (optional but helpful)
    @app.route('/api/engine_status')
    def engine_status():
        if app.rag_engine:
            # You might want a more detailed status method on your engine
            return jsonify({"status": "initialized", "params": app.rag_engine.get_parameters()})
        else:
            return jsonify({"status": "not_initialized"})
            
    return app