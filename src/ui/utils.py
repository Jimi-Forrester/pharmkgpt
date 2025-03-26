import gradio as gr
import html
from src.ui.config import CHOICES, MODELS_REQUIRING_KEY
import json
from flask import Flask
from src.rag import RAGEngine
import os
import socket 
import time 
from threading import Thread 

os.environ["GRADIO_ANALYTICS"] = "False"


def json_to_kg_list(json):
    nodes_list = str(json['nodes']).replace("'id'", 'id').replace("'label'", 'name').replace("'color'", 'color').replace("'name'", 'name').replace("'title'", 'title').replace("'size'", 'size')
    edges_list = str(json['edges']).replace("'from'", 'from').replace("'to'", 'to').replace("'label'", 'name').replace("'color'", 'color').replace("'name'", 'name').replace("'title'", 'title').replace("'size'", 'size')
    return nodes_list, edges_list

def kg_to_visjs(json):
  nodes_list, edges_list = json_to_kg_list(json)
  html_content = r"""
  <!DOCTYPE html> 
  <html> 
  
  <head> 
    <meta charset="utf-8"> 
    <title>Vis.js  Knowledge Graph (Neo4j-like Style)</title> 
    <!-- 引入 vis.js  的 CSS 和 JS --> 
    <script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/vis/4.21.0/vis.min.js"></script>  
    <link href="https://cdnjs.cloudflare.com/ajax/libs/vis/4.21.0/vis.min.css"  rel="stylesheet" type="text/css" /> 
    <style> 
      /* 页面整体样式 */ 
      body { 
        margin: 0; 
        padding: 0; 
        background-color: #ffffff; /* 白色背景 */ 
        font-family: "Helvetica Neue", Helvetica, Arial, sans-serif; 
      } 
  
      #network { 
        width: 100%; 
        height: 100vh; 
        border: none; 
      } 
    </style> 
  </head> 
  
  <body> 
    <div id="network"></div> 
    <script type="text/javascript"> 
      window.onload  = function () { 
        // 示例节点，分三类：粉色、绿色、蓝色 
        // 你可以根据实际数据，为节点加上 group 或 color"""
  html_content +=f""" 
        var nodes = new vis.DataSet({nodes_list}); 
        var edges = new vis.DataSet({edges_list}); 
  """

  html_content +=r"""// vis.js  配置 
        var options = { 
          nodes: {
            shape: 'dot', 
            borderWidth: 0,
            scaling: { min: 10, max: 50 }
          }, 
          edges: { 
            width: 2, 
            color: { 
              //color: '#999999', 
              //highlight: '#666666' 
              inherit: 'both'
            }, 
            arrows: { 
              to: { enabled: false, scaleFactor: 0.8 }  // 在目标节点处显示箭头 
            }, 
            font: { 
              color: '#333333', 
              size: 14, 
              align: 'horizontal' 
            }, 
            smooth: { 
              enabled: false, 
              type: 'dynamic' 
            },
            scaling: {
              min:1,
              max:3
            }
          }, 
          physics: { 
            enabled: true, 
            stabilization: {iterations:500}, 
            barnesHut: { 
              gravitationalConstant: -3500, 
              springConstant: 0.1, 
              springLength: 120,
              damping: 0.2,
            } 
          }, 
          interaction: { 
            hover: true, 
            tooltipDelay: 200, 
            dragNodes: true, 
            zoomView: true 
          }, 
          layout: { 
            improvedLayout: true,
            hierarchical: false
          } 
        }; 
  
        var container = document.getElementById('network');  
        var data = { nodes: nodes, edges: edges }; 
        var network = new vis.Network(container, data, options); 
      }; 
    </script> 

  """
  return html_content
def chat_reset():
    return INITIAL_CHATBOT, INITIAL_HTML, INITIAL_MESSAGE_BOX

INITIAL_CHATBOT = []
INITIAL_HTML = ""
INITIAL_MESSAGE_BOX = ""
LASTET_MESSAGE = ""
MODELS = ["DeepSeek-R1","Gemini",]
choices = ["Drugs", "Diseases", "Gene", "Protein", "Metabolite", "SNP", "Clinical"]

def query(question):
    import requests
    import json

    # Flask 服务器地址
    url = "http://127.0.0.1:5000/query"
    # 发送的 JSON 数据
    data = {"question": question}

    # 发送 POST 请求
    response = requests.post(url, json=data)

    # 解析并打印返回结果
    if response.status_code == 200:
        print("✅ 成功！返回结果：")
        print(json.dumps(response.json(), indent=4, ensure_ascii=False))
        return response.json()
    else:
        print(f"❌ 失败！错误码: {response.status_code}")
        print(response.text)
        return {"Answer":"Busy now. Please try again later."}
    
# 全选逻辑函数 
def toggle_select_all(select_all, checkbox_group):
    if select_all and len(checkbox_group) != len(choices):
        return choices
    elif not select_all and len(checkbox_group) == len(choices):
        return []
    return checkbox_group

def toggle_select_all_reverse(select_list):
    return True if len(select_list) == len(choices) else False

def user_msg(message, chat_history):
    # 仅将用户消息加入历史记录，并立即清空输入框
    if message != "":
        chat_history.append([message,  None])  # None表示机器人未回复 
        return "", chat_history  # 返回空字符串清空输入框
    else:
        gr.Warning("Please enter a message.")
        return "", chat_history

def load_and_display(output):
    # 生成动态可折叠面板的HTML内容 
    accordion_html = "<div style='width:100%; margin:10px 0;'>"
    for key, ref in output['Context'].items(): 
        title = ref['title']
        abstract = ref['abstract']
        score = ref['score']
        pmid = ref['pmid']
        accordion_html += f"""
        <details style='margin:10px 0; border:1px solid #FFFFFF; border-radius:5px;'>
            <summary style='cursor:pointer; padding:10px; background:#FFFFFF;'>
                {title}<a href="https://pubmed.ncbi.nlm.nih.gov/{pmid}/" target="_blank">[PMID: {pmid}]</a>
            </summary>
            <div style='padding:15px; background:white;'>
                <b>Relevance score:</b> {round(score, 2)}
            </div>
            <div style='padding:15px; background:white;'>
                {abstract.replace('\n',  '<br>')}
            </div>
        </details>
        """
    accordion_html += "</div></body></html> "
    return accordion_html 


def respond(chat_history):
    message = chat_history[-1][0]  # 获取用户最新消息
    # 调用RAG生成回复 
    output = query(message)
    # with open("./src/ui/output.json", "r") as f:
    #     output = json.load(f)
    #将回复加入历史记录
    html_content = kg_to_visjs(output['KG'])
    chat_history[-1][1] = output['Answer']
    escaped_html = html.escape(html_content, quote=True)
    iframe_code = f"""
    <iframe srcdoc="{escaped_html}" style='width:100%; height:400px; border:none;'></iframe>
    """

    iframe_code += load_and_display(output)
    return chat_history, iframe_code

    
def interaction_true():
    return gr.update(interactive=True),gr.update(interactive=True),gr.update(interactive=True),gr.update(interactive=True),gr.update(interactive=True),gr.update(interactive=True)
def interaction_false():
    return gr.update(interactive=False),gr.update(interactive=False),gr.update(interactive=False),gr.update(interactive=False),gr.update(interactive=False),gr.update(interactive=False)

import time
import socket
from threading import Thread
from flask import Flask, request, jsonify
import threading

_app_instance = None
_server_thread = None
# _app_lock = threading.Lock()

def get_or_create_app(model_type='Qwen2.5',api_key=None, top_k=5, hops=1):
    global _app_instance
    if _app_instance is None:
        print("Creating Flask app for the first time...")
        app = Flask("kg2rag")  # 创建 Flask 应用
        app.rag_engine = RAGEngine(model_type=model_type,api_key=api_key,top_k=top_k, hops=hops)  # 绑定 RAGEngine 到 app 实例
        # 注册路由
        from src.routes import register_routes
        register_routes(app)
        _app_instance = app
    return _app_instance
def is_port_open(port):
    sock = socket.socket(socket.AF_INET,  socket.SOCK_STREAM)
    try:
        sock.connect(('127.0.0.1',  port))
        return True
    except ConnectionRefusedError:
        return False 
    finally:
        sock.close()
def start_server_if_needed(model_type, api_key, top_k, hops):
    global _server_thread, _app_instance
    # with _app_lock:
    if _server_thread is None or not _server_thread.is_alive():
        print("Starting Flask server...")
        app = get_or_create_app(model_type=model_type,api_key=api_key,top_k=top_k, hops=hops)

        # 启动Flask服务线程 
        _server_thread = Thread(target=app.run,  kwargs={'debug': False, 'port': 5000, 'use_reloader': False})
        _server_thread.daemon  = True  # 设置为守护线程，主线程退出时自动结束 
        _server_thread.start()

        # 等待直到端口开放（表示服务启动完成）
        while not is_port_open(5000):
            time.sleep(0.5)
        print("Flask server started.")
    else:
        print("Flask server already running.")
          
def parameters_embedding_live_update(model_type, api_key, top_k, hops):
    port=5000
    host='127.0.0.1'
    # 1. 确保服务器正在运行
    start_server_if_needed(model_type, api_key, top_k, hops)
    # 2. 构造要发送的数据
    payload = {
        "model_type": model_type,
        "api_key": api_key,
        "top_k": top_k,
        "hops": hops
    }
    # 3. 发送 POST 请求到 /update_config 端点
    update_url = f'http://{host}:{port}/update_config'
    print(f"Sending update request to {update_url} with payload: {payload}")
    try:
        import requests
        response = requests.post(update_url, json=payload, timeout=30) # 增加超时以等待可能的模型加载
        response.raise_for_status() # 如果状态码不是 2xx，则抛出异常
        print("Update request successful:")
        print(response.json())
    except requests.exceptions.RequestException as e:
        print(f"Error sending update request: {e}")
        if e.response is not None:
            print("Server response:", e.response.text)
    except Exception as e:
        print(f"An unexpected error occurred during update: {e}")

def api_check(selected_model, api_key):
    print(f"Model: {selected_model}")
    # 只有在需要 API Key 的模型被选中时，才处理或验证 API Key
    if selected_model in MODELS_REQUIRING_KEY:
        if not api_key:
            raise gr.Error(f"API Key is required for {selected_model}")
        else:
            print(f"API Key Provided: {'*' * len(api_key)}")
            return gr.update(visible=True)
    else:
        print("API Key: Not required for this model.")
        return gr.update(visible=True)


# --- 控制 API Key 输入框可见性的函数 ---
def update_api_key_visibility(selected_model):
    """根据选择的模型更新 API Key 输入框的可见性"""
    if selected_model in MODELS_REQUIRING_KEY:
        # 如果选中的模型需要 Key，则返回一个更新对象使 API Key 输入框可见
        return gr.update(visible=True)
    else:
        # 否则，使其不可见
        return gr.update(visible=False)