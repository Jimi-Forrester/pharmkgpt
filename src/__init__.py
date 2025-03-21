from flask import Flask
from src.rag import RAGEngine




def create_app():
    app = Flask(__name__)  # 创建 Flask 应用
    app.rag_engine = RAGEngine()  # 绑定 RAGEngine 到 app 实例

    # 注册路由
    from src.routes import register_routes
    register_routes(app)

    return app
