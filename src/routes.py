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
    
@bp.route('/update_config', methods=['POST'])
def update_engine_config():
    from flask import current_app
    data = request.get_json()
    rag_engine = current_app.rag_engine
    if not data:
        return jsonify({"error": "Invalid JSON payload"}), 400
    # 从请求中获取参数
    model_type = data.get('model_type')
    api_key = data.get('api_key') # 注意处理 null/None
    top_k = data.get('top_k')
    hops = data.get('hops')
    
    try:
        # 调用 RAGEngine 的更新方法
        rag_engine.update_config(
            model_type=model_type,
            api_key=api_key,
            top_k=top_k,
            hops=hops
        )
        return jsonify({"message": "Configuration updated successfully", "new_params": rag_engine.get_parameters()}), 200
    except Exception as e:
        # 处理更新过程中可能发生的错误 (例如模型加载失败)
        print(f"Error updating RAGEngine config: {e}")
        return jsonify({"error": f"Failed to update configuration: {e}"}), 500
@bp.route('/current_params')
def get_current_params_route():
    from flask import current_app
    rag_engine = current_app.rag_engine
    return jsonify(rag_engine.get_parameters())

def register_routes(app):
    app.register_blueprint(bp)
