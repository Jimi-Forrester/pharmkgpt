from flask import Blueprint, request, jsonify

bp = Blueprint("api", __name__)

@bp.route("/query", methods=["POST"])
def query_rag():
    from flask import current_app  # 使用 current_app 访问 Flask 实例
    
    rag_engine = current_app.rag_engine
    data = request.get_json()
    if not data or "question" not in data:
        return jsonify({"error": "Missing question"}), 400
    question = data["question"]
    try:
        result = rag_engine.query(question)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def register_routes(app):
    app.register_blueprint(bp)
