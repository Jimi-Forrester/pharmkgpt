import gradio as gr
import html
from src.ui.config import *
import os

os.environ["GRADIO_ANALYTICS"] = "False"

def json_to_kg_list(json):
    nodes_list = str(json['nodes']).replace("'id'", 'id').replace("'label'", 'name').replace("'color'", 'colors').replace("'name'", 'name').replace("'title'", 'title').replace("'size'", 'size').replace("'group'", 'group')
    edges_list = str(json['edges']).replace("'from'", 'from').replace("'to'", 'to').replace("'label'", 'name').replace("'color'", 'colors').replace("'name'", 'name').replace("'title'", 'title').replace("'size'", 'size').replace("'group'", 'group')
    return nodes_list, edges_list
en_color = {
    'article': {"background": '#A7CBFC', "border": '#1B3B6F'},  # 赛博蓝/深海蓝
    'chemical': {"background": '#35A29F', "border": '#1F6F78'},  # 青绿/深青蓝
    'disease': {"background": '#FF3864', "border": '#A30052'},  # 霓虹红/暗紫红 (主色)
    'gene': {"background": '#FFCE4F', "border": '#D48C00'},  # 明亮黄/金黄
    'metabolite': {"background": '#E457A6', "border": '#9C1B6C'},  # 霓虹粉/暗粉紫
    'pathway': {"background": '#8A5CF6', "border": '#4E2A84'},  # 科技紫/深紫
    'processes': {"background": '#6BE3C3', "border": '#2A9D8F'},  # 赛博绿/深绿
    'protein': {"background": '#1E88E5', "border": '#0D47A1'},  # 电蓝/深蓝
    'region': {"background": '#9E7A65', "border": '#9E7A65'},
}
def kg_to_visjs(json_input): # 修改函数签名以接收输入
    nodes_list_json, edges_list_json = json_to_kg_list(json_input) # 调用转换函数
    try:
        group_counts = {}
        for node in json_input['nodes']:
            group_name = node['group'] # 获取当前节点的 group 名称
            # 使用字典的 .get(key, default) 方法
            # 如果 group_name 已经在字典中，返回它的当前计数值，否则返回 0
            # 然后将计数值加 1，并更新（或添加）到字典中
            group_counts[group_name] = group_counts.get(group_name, 0) + 1
    except:
        print("Error: Unable to count nodes list.")
    # --- 生成图例的 HTML ---
    legend_html_items = ""
    for key, colors in en_color.items():
        background_color = colors['background']
        text_color = '#ffffff'
        try:
            legend_html_items += f"""
            <div class="legend-item" style="background-color: {background_color}; border:none; color: {text_color};">
                {key.capitalize() + " (" + str(group_counts[key]) + ")"}
            </div>
            """
        except:
            pass
    # --- 图例HTML结束 ---

    html_content = rf"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Vis.js Knowledge Graph with Legend</title>
        <!-- 引入 vis.js 的 CSS 和 JS -->
        <script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/vis/4.21.0/vis.min.js"></script>
        <link href="https://cdnjs.cloudflare.com/ajax/libs/vis/4.21.0/vis.min.css" rel="stylesheet" type="text/css" />
        <style>
          /* 页面整体样式 */
          body {{
            margin: 0;
            padding: 0;
            background-color: #ffffff; /* 白色背景 */
            font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
            display: flex; /* 使用 flex布局，方便并列 */
            height: 100vh; /* 视口高度 */
            overflow: hidden; /* 防止滚动条 */
          }}
          #network {{
            /* width: calc(100% - 200px); /* 减去图例宽度 */
            flex-grow: 1; /* 让网络图占据剩余空间 */
            height: 100%; /* 占满父容器高度 */
            border: none;
          }}

          /* 图例样式 */
          #legend {{
            width: 120px;   /* Keep adaptive width */
            background-color: #fff; /* Slightly different background maybe */
            border-left: none; /* No border */
            padding: 15px;
            height: 100%;          /* MODIFIED: Make legend full height */
            box-sizing: border-box; /* ADD: Include padding/border in height calculation */
            overflow-y: auto;      /* Keep vertical scroll if needed */
            display: flex;
            flex-direction: column; /* MODIFIED: Stack title and items vertically */
            flex-wrap: nowrap;      /* MODIFIED: Prevent wrapping of title/items container */
            gap: 8px;             /* Adjust gap */
            flex-shrink: 0;       /* ADD: Prevent legend from shrinking */
          }}
          /* ADD a container for the legend items to apply wrap */
          .legend-items-container {{
              display: flex;
              flex-wrap: wrap;
              gap: 8px; /* Spacing between individual legend items */
              justify-content: center;
          }}

          #legend h3 {{
             margin-top: 0;
             margin-bottom: 5px; /* Adjust spacing */
             text-align: center;
             font-size: 16px;
             color: #333;
             width: 100%; /* Ensure header takes full width */
             flex-shrink: 0; /* Prevent header from shrinking */
          }}

          .legend-item {{
            padding: 6px 12px;
            border-radius: 15px;
            text-align: center;
            font-size: 12px;
            font-weight: bold;
            cursor: default;
            flex-grow: 0; /* Don't allow items to grow */
            flex-shrink: 0; /* Don't allow items to shrink */
             /* min-width, background-color, border, color set via inline style */
          }}
      </style>
    </head>

    <body>
      <div id="network"></div>

    <!-- 图例容器 -->
      <div id="legend">
        <!-- ADDED Wrapper Div -->
        <div class="legend-items-container">
            {legend_html_items}
        </div>
      </div>

      <script type="text/javascript">
        window.onload = function () {{
          // 节点和边数据从 Python 传入
          var nodes = new vis.DataSet({nodes_list_json});
          var edges = new vis.DataSet({edges_list_json});

          // vis.js 配置 (保持不变)
          var options = {{
            nodes: {{
                shape: 'dot',
                borderWidth: 0,
                //scaling: {{ min: 25, max: 65 }}
            }},
            edges: {{
              width: 2,
              color: {{
                inherit: 'both' // 边的颜色可以继承自节点，或设为固定颜色
              }},
              arrows: {{
                to: {{ enabled: false, scaleFactor: 0.8 }} // 保持箭头不可见
              }},
              font: {{
                color: '#333333',
                size: 14, // 边标签字体大小
                align: 'horizontal' // 标签沿边居中
              }},
              smooth: {{
                enabled: false // 禁用平滑曲线可能性能更好
              }},
              scaling: {{
                min:1,
                max:3
              }}
            }},
            groups: {{
                    article: {{ color:'#A7CBFC' }},
                    chemical: {{ color:'#35A29F' }},
                    disease:  {{ color:'#FF3864' }},
                    gene:     {{ color:'#FFCE4F'}},
                    metabolite:{{ color:'#E457A6'}},
                    pathway:  {{ color:'#8A5CF6'}},
                    processes:{{ color:'#6BE3C3'}},
                    protein:  {{ color:'#1E88E5'}},
                    region:   {{ color:'#9E7A65'}}
                }},
            physics: {{
              enabled: true,
              stabilization: {{iterations:1000}}, // 减少稳定迭代次数可能加快加载
              barnesHut: {{
                gravitationalConstant: -8000, // 增加斥力
                springConstant: 0.05,
                springLength: 300, // 增加弹簧长度
                damping: 0.2,
                centralGravity: 0.1,
                avoidOverlap: 1 // 增加避免重叠因子
              }}
            }},
            interaction: {{
              hover: true,
              tooltipDelay: 200,
              dragNodes: true,
              zoomView: true,
            }},
            layout: {{
              improvedLayout: true,
              hierarchical: false
            }}
          }};

          var container = document.getElementById('network');
          var data = {{ nodes: nodes, edges: edges }};
          var network = new vis.Network(container, data, options);
        }};
      </script>

    </body>
    </html>
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
            <summary style='cursor:pointer; padding:10px; background:#FFFFFF;font-weight:700;'>
                {title}<a href="https://pubmed.ncbi.nlm.nih.gov/{pmid}/" target="_blank">[PMID: {pmid}]</a>
            </summary>
            <div style='padding-left:15px; background:white;'>
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
    
        chat_history[-1][1] = "<div class='loading-text'>Loading</div>"  # 更新用户消息
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
    model_type_map = {
        'MindGPT': 'gemma3',
        'Gemini': 'gemini',
        'OpenAI': 'openai',
    }
    if model_type in model_type_map:
        model_type = model_type_map[model_type]
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
        response = requests.post(update_url, json=payload, timeout=90) # 增加超时以等待可能的模型加载
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
def get_parameter():
    import requests
    import json 

    # Flask 服务器地址
    url = F"{BASE_URL}/status"
    try:
    # 发送 GET 请求
        response = requests.get(url, timeout=60)
        # 解析并打印返回结果
        response.raise_for_status() # 如果状态码不是 2xx，则抛出异常
        res = response.json()
        if res['configured'] == True:
            return res
        elif res['configured'] == False:
            return None
            
    except requests.exceptions.RequestException as e:
         print(f"请求后端时发生错误: {e}")
         return "REQUEST_ERROR" # 其他请求错误
    except requests.exceptions.ConnectionError:
         return "CONNECTION_ERROR" # 特殊标记表示连接失败
    except Exception as e:
        print(f"处理后端响应时发生未知错误: {e}")
        return "UNKNOWN_ERROR" # 未知错误
def model_type_converter(model_type):
    model_type_map = {
        'gemma3': 'MindGPT',
        'openai': 'OpenAI',
    }
    if model_type in model_type_map:
        model_type = model_type_map[model_type]
        return model_type
    else:
        return model_type
def update_placeholder_in_load():
    status = get_parameter()
    if status:
        if status == "REQUEST_ERROR":
            return gr.update(placeholder="<p style='font-size:18px; font-weight:bold; color:red'>Server is not available, please check the server status.</p>"), gr.update(value=False), gr.update(value=MODELS[0]), gr.update(value=5)
        elif status == "CONNECTION_ERROR":
            return gr.update(placeholder="<p style='font-size:18px; font-weight:bold; color:red'>Server is not available now, please check the server status.</p>"), gr.update(value=False), gr.update(value=MODELS[0]), gr.update(value=5)
        elif status == "UNKNOWN_ERROR":
            return gr.update(placeholder="<p style='font-size:18px; font-weight:bold; color:red'>Server error, please check the server status.</p>"), gr.update(value=False), gr.update(value=MODELS[0]), gr.update(value=5)
        else:
            return gr.update(placeholder="<p style='font-size:18px; font-weight:bold'>You can ask questions.</p>"), gr.update(value=True), gr.update(value=model_type_converter(status['params']['model_type'])), gr.update(value=status['params']['top_k'])
    else:
        return gr.update(placeholder="<p style='font-size:18px; font-weight:bold; color:red'>Set parameters before Chat.</p>"), gr.update(value=False), gr.update(value=MODELS[0]), gr.update(value=5)
def get_parameter_in_send():
    res = get_parameter()
    if res:
        if res == "REQUEST_ERROR":
            raise(gr.Error("Server is not available, please check the server status."))
        elif res == "CONNECTION_ERROR":
            raise(gr.Error("Server is not available, please check the server status."))
        elif res == "UNKNOWN_ERROR":
            raise(gr.Error("UNKNOWN_ERROR"))
        else:
            pass
            # return gr.update(value=True)
    else:
        raise(gr.Error("Your parameters lost, please set parameters again"))

import matplotlib.pyplot  as plt 
import pandas as pd 
from matplotlib.colors  import LinearSegmentedColormap 
 
def plot_interactive_hbar():
    # 数据定义 
    cont = {'disease': 7895,
            'gene': 19549,
            'article': 153060,
            'chemical': 10295,
            'protein': 7062,
            'region': 3419,
            'processes': 5170,
            'pathway': 3128,
            'metabolite': 2170,}
    
    # 数据处理 
    df = pd.DataFrame(list(cont.items()),  columns=['category', 'count'])
    df = df.sort_values('count',  ascending=True)
    
    # 自定义渐变颜色（参考原Plotly的绿-蓝渐变）
    colors = [(16/255, 185/255, 129/255), (56/255, 173/255, 198/255), (59/255, 130/255, 246/255)]
    cmap = LinearSegmentedColormap.from_list("custom_gradient",  colors, N=256)
    norm = plt.Normalize(df['count'].min(), df['count'].max())
    
    # 创建图表 
    fig, ax = plt.subplots(figsize=(8,  7))  # 增加画布高度 
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

def toggle_sidebar(current_visibility):
    """
    Updates the visibility of the sidebar column and the button text/icon.

    Args:
        current_visibility (bool): The current value of the sidebar_visible state.

    Returns:
        tuple: Updates for (sidebar_column, toggle_button, sidebar_visible_state)
    """
    new_visibility = not current_visibility
    button_text = "〈 Hide" if new_visibility else "〉 Show" # Or use icons like "<", ">"
    print(f"Toggling sidebar visibility to: {new_visibility}")
    return (
        gr.update(visible=new_visibility), # Update sidebar column visibility
        gr.update(value=button_text),     # Update button text/icon
        new_visibility                    # Update the state variable itself
    )