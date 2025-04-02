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
              stabilization: {iterations:1000}, 
              barnesHut: { 
                gravitationalConstant: -20000, 
                springConstant: 0.05, 
                springLength: 300,
                damping: 0.2,
                centralGravity: 0.1,
                avoidOverlap: 1
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
    # url = "http://127.0.0.1:5000/query"
    url = "http://192.168.110.64:5000/api/query"
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
def query_stream(question):
    import requests
    import json

    # Flask 服务器地址
    # url = "http://127.0.0.1:5000/query"
    url = "http://192.168.110.64:5000/api/query"
    # 发送的 JSON 数据
    data = {"question": question}

    # 发送 POST 请求
    response = requests.post(url, json=data, stream=True)
    # 解析返回的流式数据
    for line in response.iter_lines():
        if line:
            try:
                # 解析 JSON 数据
                progress_json = json.loads(line.decode('utf-8'))
                yield progress_json
            except json.JSONDecodeError as e:
                print(f"JSON 解析错误: {e}")
                continue

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
    context_data = output['Context']
    sorted_items = sorted(context_data.items(),  key=lambda item: item[1]["score"], reverse=True)
    sorted_context = dict(sorted_items) 
    for key, ref in sorted_context.items(): 
        title = ref['title']
        abstract = ref['abstract'].replace('\n',  '<br>')
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
                {abstract}
            </div>
        </details>
        """
    accordion_html += "</div></body></html> "
    return accordion_html 


def respond(chat_history):
    message = chat_history[-1][0]  # 获取用户最新消息
    # 调用RAG生成回复 
    try:
        output = query(message)
    except Exception as e:
        output = {"Answer":"Busy now. Please try again later."}
    # with open("./src/ui/output.json", "r") as f:
    #     output = json.load(f)
    #将回复加入历史记录
    chat_history[-1][1] = output['Answer']
    try:
        html_content = kg_to_visjs(output['KG'])
        escaped_html = html.escape(html_content, quote=True)
        iframe_code = f"""
        <iframe srcdoc="{escaped_html}" style='width:100%; height:40vh; border:none;'></iframe>
        """
    except Exception as e:
        print(e)
        html_content = ""
        iframe_code = ""
      
    try:
        iframe_code += load_and_display(output)
    except Exception as e:
        print(e)
    return chat_history, iframe_code

def yield_respond(msg, chat_history):

    chat_history.append([msg, "Loading..."])  # 更新用户消息
    yield "", chat_history, "<p>Loading...</p>" 
    # 调用RAG生成回复 
    for progress_json in query_stream(msg):
        if progress_json['type'] == 'progress':
            chat_history[-1] = [msg, progress_json['message']]  # 更新聊天记录
            yield "", chat_history, "<p>Loading...</p>" # 返回更新后的聊天记录
        elif progress_json['type'] == 'result':
            output = progress_json['data']
            chat_history[-1][1] = output['Answer']  # 更新聊天记录
            try:
                html_content = kg_to_visjs(output['KG'])
                escaped_html = html.escape(html_content, quote=True)
                iframe_code = f"""
                <iframe srcdoc="{escaped_html}" style='width:100%; height:40vh; border:none;'></iframe>
                """
            except Exception as e:
                print(e)
                html_content = ""
                iframe_code = ""
            try:
                iframe_code += load_and_display(output)
            except Exception as e:
                print(e)
            yield "", chat_history, iframe_code  # 返回更新后的聊天记录和HTML内容

def interaction_true():
    return gr.update(interactive=True),gr.update(interactive=True),gr.update(interactive=True),gr.update(interactive=True),gr.update(interactive=True),gr.update(interactive=True)
def interaction_false():
    return gr.update(interactive=False),gr.update(interactive=False),gr.update(interactive=False),gr.update(interactive=False),gr.update(interactive=False),gr.update(interactive=False)

def parameters_embedding_live_update(model_type='gemma3', api_key=None, top_k=5, hops=1):
    port=5000
    # host='127.0.0.1'
    host='192.168.110.64'
    # 1. 确保服务器正在运行
    # start_server_if_needed(model_type, api_key, top_k, hops)
    # 2. 构造要发送的数据
    payload = {
        "model_type": model_type,
        "api_key": api_key,
        "top_k": top_k,
        "hops": hops
    }
    # 3. 发送 POST 请求到 /update_config 端点
    update_url = f'http://{host}:{port}/api/update_config'
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

def select_base_change(select_base):
    if "File(s)" in select_base:
        return gr.update(visible=True)
    else:
        return gr.update(visible=False)
    
def upload_to_database(file_name):
    return file_name
def process_file(file_obj, init_choices, progress=gr.Progress()):
    import time
    # 初始化处理进度 
    progress(0, desc="开始处理文件...")
    
    # 模拟处理过程（替换为实际处理逻辑）
    for i in progress.tqdm(range(100)): 
        time.sleep(0.01) 
        if i == 50:
            progress(0.5, desc="处理中...")
    
    # 上传到Dropbox 
    progress(0.8, desc="上传...")
    file_name = upload_to_database('uploaded_file') 
    
    # 更新可选文件列表 
    return gr.update(choices=init_choices+[file_name], visible=True)
def update_state(model_selector, top_k, hops):
    """
    更新全局状态的函数。
    Args:
        model_selector:  gr.Dropdown 组件的值。
        top_k: gr.Slider 组件的值。
        hops: gr.Slider 组件的值。
        current_model_selector: 前一个状态的 current_model_selector。  初始值为 None.
        current_top_k: 前一个状态的 current_top_k。 初始值为 None.
        current_hops: 前一个状态的 current_hops。 初始值为 None.
    Returns:
        Tuple[str, int, int, str, int, int]:  新的 current_model_selector, current_top_k, current_hops 状态，以及原封不动地返回输入值。
    """
    new_model_selector = model_selector
    new_top_k = top_k
    new_hops = hops
    # 打印更新信息，方便调试
    print(f"Updating state:")
    print(f"new model: {new_model_selector}")
    print(f"new top_k: {new_top_k}")
    print(f"new hops: {new_hops}")
    return new_model_selector, new_top_k, new_hops
def are_parameters_same(model_selector, top_k, hops, current_model_selector, current_top_k, current_hops):
    """
    比较参数是否相同。  这个函数仅仅用于获取参数值， 不做任何更新操作。
    Args:
        model_selector:  gr.Dropdown 组件的值。
        top_k: gr.Slider 组件的值。
        hops: gr.Slider 组件的值。
        current_model_selector:  current_model_selector 的值
        current_top_k:  current_top_k 的值
        current_hops:  current_hops 的值
    Returns:
        bool:  如果所有参数都相同，返回 True，否则返回 False
    """
    if (model_selector == current_model_selector and
        top_k == current_top_k and
        hops == current_hops):
        return gr.update(disabled=True)
    else:
        return gr.update(disabled=False)
def get_init_parameter():
    import requests
    import json

    # Flask 服务器地址
    url = "http://192.168.110.64:5000/api/current_params"
    # 发送 GET 请求
    response = requests.get(url)
    # 解析并打印返回结果
    if response.status_code == 200:
        res = response.json()
        return gr.State(value=res['model_type']), gr.State(value=res['top_k']), gr.State(value=res['hops'])
    else:
        return gr.State(value=MODELS[0]), gr.State(value=5), gr.State(value=1)