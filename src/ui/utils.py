import gradio as gr
import html
from src.ui.config import *
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

def query_stream(question):
    import requests
    import json

    # Flask 服务器地址
    url = F"{BASE_URL}/query"
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
    if select_all and len(checkbox_group) != len(CHOICES):
        return CHOICES
    elif not select_all and len(checkbox_group) == len(CHOICES):
        return []
    return checkbox_group

def toggle_select_all_reverse(select_list):
    return True if len(select_list) == len(CHOICES) else False

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
                <b>Relevance score:</b> {score:.3f}
            </div>
            <div style='padding:15px; background:white;'>
                {abstract}
            </div>
        </details>
        """
    accordion_html += "</div></body></html> "
    return accordion_html 

def user_msg(message, chat_history):
    # 仅将用户消息加入历史记录，并立即清空输入框
    if message != "":
        chat_history.append([message,  None])  # None表示机器人未回复 
        return "", chat_history  # 返回空字符串清空输入框
    else:
        gr.Warning("Please enter a message.")
        return "", chat_history
def yield_respond(chat_history):
    try:
        msg = chat_history[-1][0]  # 更新用户消息
    
        chat_history[-1][1] = "<div class='loading-text'>Loading...</div>"  # 更新用户消息
        yield chat_history, gr.State(None)
        # 调用RAG生成回复 
        for progress_json in query_stream(msg):
            if progress_json['type'] == 'progress':
                progress_msg = progress_json['message']
                progress_list = progress_msg.split("\n")
                if len(progress_list) > 1:
                    progress_content = ''
                    last_progress_msg = progress_list.pop(-1)
                    for line in progress_list:
                        if line != "":
                            progress_content += f"<div class='loaded-text'>{line}</div>"
                    progress_content += f"<div class='loading-text'>{last_progress_msg}</div>"
                else:
                    progress_content = ''
                    last_progress_msg = progress_list[0]
                    progress_content = f"<div class='loading-text'>{last_progress_msg}</div>"
                    
                chat_history[-1] = [msg, progress_content]  # 更新聊天记录
                yield chat_history,  gr.State(None)# 返回更新后的聊天记录
            elif progress_json['type'] == 'result':
                query_result = progress_json['data']
                yield chat_history,  gr.State(query_result)# 返回更新后的聊天记录
    except Exception as e:
        print(e)
        yield chat_history,  gr.State(None)  # 返回更新后的聊天记录
def answer_respond(chat_history, query_result):
    output = query_result.value
    if output:
        # 将回复加入历史记录
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
    else:
        # 如果没有输出，返回空值
        return chat_history, ""

def interaction_true(*args):
    # Create a list of updates, one for each argument passed
    updates = [gr.update(interactive=True) for _ in args]
    # Return as a tuple, which is standard for multiple Gradio outputs
    return tuple(updates)
def interaction_false(*args):
    # Create a list of updates, one for each argument passed
    updates = [gr.update(interactive=False) for _ in args]
    # Return as a tuple, which is standard for multiple Gradio outputs
    return tuple(updates)
def parameters_embedding_live_update(model_type='gemma3', api_key=None, top_k=5, hops=1):
    # 1. 构造要发送的数据
    if model_type == 'MindGPT':
        model_type = 'gemma3'
    payload = {
        "model_type": model_type,
        "api_key": api_key,
        "top_k": top_k,
        "hops": hops
    }
    # 2. 发送 POST 请求到 /update_config 端点
    update_url = f'{BASE_URL}/update_config'
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
    return new_model_selector, gr.update(value = new_top_k), gr.update(value = new_hops)
def are_parameters_same(model_selector, top_k, hops, current_model_selector, current_top_k, current_hops, is_parameters_set):
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
        hops == current_hops) and (is_parameters_set == True):
        return gr.update(interactive=False, value="Current parameters"), gr.update(value=current_model_selector), gr.update(current_top_k), gr.update(current_hops)
    else:
        return gr.update(interactive=True, value="Set parameters"), gr.update(value=current_model_selector), gr.update(current_top_k), gr.update(current_hops)
def get_init_parameter():
    import requests
    import json

    # Flask 服务器地址
    url = F"{BASE_URL}/current_params"
    # 发送 GET 请求
    response = requests.get(url)
    # 解析并打印返回结果
    if response.status_code == 200:
        res = response.json()
        if res['model_type'] and res['top_k'] and res['hops']:
            return gr.State(value=res['model_type']), gr.State(value=res['top_k']), gr.State(value=res['hops'])
        else:
            parameters_embedding_live_update()
            return gr.State(value=MODELS[0]), gr.State(value=5), gr.State(value=1)
        
    else:
        return gr.State(value=MODELS[0]), gr.State(value=5), gr.State(value=1)
    # return gr.State(value=MODELS[0]), gr.State(value=5), gr.State(value=1)


import matplotlib.pyplot  as plt 
import pandas as pd 
from matplotlib.colors  import LinearSegmentedColormap 
 
def plot_interactive_hbar():
    # 数据定义 
    cont = {
        'disease': 196769,
        'gene': 8802,
        'chemical': 39106,
        'pubmed': 46835,
        'metabolite': 156,
        'protein': 435,
        'processes': 225 
    }
    
    # 数据处理 
    df = pd.DataFrame(list(cont.items()),  columns=['category', 'count'])
    df = df.sort_values('count',  ascending=True)
    
    # 自定义渐变颜色（参考原Plotly的绿-蓝渐变）
    colors = [(16/255, 185/255, 129/255), (56/255, 173/255, 198/255), (59/255, 130/255, 246/255)]
    cmap = LinearSegmentedColormap.from_list("custom_gradient",  colors, N=256)
    norm = plt.Normalize(df['count'].min(), df['count'].max())
    
    # 创建图表 
    fig, ax = plt.subplots(figsize=(8,  8))  # 增加画布高度 
    bars = ax.barh(df['category'],  df['count'], color=cmap(norm(df['count'])))
    
    # 标签设置 
    for bar in bars:
        width = bar.get_width() 
        ax.text(width*1.02,  bar.get_y()  + bar.get_height()/2,  
                f'{width:,}', 
                ha='left', va='center', 
                fontsize=18, color='#000000')  # 灰色文字更清晰 
    
    # 坐标轴优化 
    ax.set_xscale('log') 
    ax.xaxis.set_visible(False) 
    ax.grid(False)  # 关闭网格线 
    ax.set_frame_on(False)  # 关闭边框 
    # 解决标签显示问题
    plt.subplots_adjust(left=0.2)   # 左侧留出更多空间 
    ax.tick_params(axis='y',  labelsize=20, pad=4)  # 调整字体和间距 
    
    # 透明背景 
    ax.set_facecolor('none') 
    fig.patch.set_facecolor('none') 
    
    return fig
