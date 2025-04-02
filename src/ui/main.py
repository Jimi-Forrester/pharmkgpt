from src.ui.utils import *
from src.ui.theme import Kotaemon

import gradio as gr
from src.ui.config import *
import os

os.environ["GRADIO_ANALYTICS"] = "False"


def ui():
    base_dir = os.path.dirname(os.path.abspath(__file__)) 
    with open(f"{base_dir}/gradio.css", "r", encoding="utf-8") as f:
        custom_css = f.read()

    with gr.Blocks(theme=Kotaemon(),title="MindResilience",css=custom_css,fill_width=True) as demo:
    # with gr.Tab("KG2RAG",elem_id="chat-tab"):
        with gr.Row():
            # 左侧设置面板 
            with gr.Column(scale=1,elem_id="left-column", min_width=260):
                gr.Markdown("# MindResilience",height="15vh")
                new_chat = gr.Button(
                        value="New chat",
                        min_width=120,
                        size="sm",
                        variant="primary",
                        elem_id="styled-btn",
                    )
                with gr.Row(visible=False) as confirm_dialog:
                    with gr.Row():
                        gr.Markdown("### Confirm to reset conversation?",)
                    confirm_btn = gr.Button("Confirm", variant="stop")
                    cancel_btn = gr.Button("Cancel")
                with gr.Accordion(label="Model parameter") as model_parameter:
                    # 定义 gr.State 组件，用于存储全局变量
                    current_model_selector, current_top_k, current_hops = get_init_parameter()
                    model_selector = gr.Dropdown(
                                    label="Model",
                                    choices=MODELS,
                                    value=MODELS[0],  # 默认选中第一个
                                    interactive=True,
                                    elem_id="model-selector",
                                )
                    initial_visibility = MODELS[0] in MODELS_REQUIRING_KEY
                    api_key_input = gr.Textbox(value=None,
                        label="API Key",
                        placeholder="Enter your API key here",
                        type="password", # 隐藏输入内容
                        visible=initial_visibility, # 设置初始可见性
                        interactive=True,
                        container=True # 让它看起来和其他组件排版一致
                    )
                    top_k = gr.Slider(minimum=1, maximum=20, value=5, step=1, label="Max Retrieved",interactive=True, elem_id="top-k")
                    hops = gr.Slider(minimum=1, maximum=5, value=1, step=1, label="Multi-hop expansion",interactive=True, elem_id="hops", visible=False)
                    set_parameters = gr.Button(
                        value="Set parameters",
                        min_width=120,
                        size="sm",
                        variant="primary",
                        elem_id="parameter-btn",
                    )
                    with gr.Row(visible=False) as confirm_set_dialog:
                        with gr.Row():
                            gr.Markdown("### Confirm to set model parameters?",)
                        confirm_set_btn = gr.Button("Confirm", variant="stop")
                        cancel_set_btn = gr.Button("Cancel")
                    waiting_text = gr.Markdown(
                        "### Waiting for model parameters to be set...",
                        visible=False, 
                    )
                with gr.Accordion(label="Knowledge base", visible=False) as knowledge_base:
                    with gr.Row():
                        select_base = gr.CheckboxGroup(choices=["PubMed", "File(s)"], value=["PubMed"], interactive=True, show_label=False)
                        file_dropdown = gr.Dropdown(
                            choices=['file1', 'file2', 'file3'], multiselect=True, interactive=True, show_label=False, visible=False)
                with gr.Accordion(label="File upload", open=False, visible=False) as file_input_section:
                    with gr.Row():
                        file_box = gr.File(file_count="single", file_types=['pdf','txt'], type="filepath", interactive=True, show_label=False, height=140)
                        progress_bar = gr.Textbox(label="处理状态", visible=False)


            with gr.Column(scale=4,elem_id="mid-column"):
                chatbot = gr.Chatbot(value=[],
                                    height="85vh",label="Chat",
                                    placeholder="You can ask questions.",
                                    show_label=False,
                                    elem_id="main-chat-bot",
                                    show_copy_button=True,
                                    likeable=False,
                                    bubble_full_width=False)
                with gr.Row(equal_height=True):
                    msg = gr.Textbox(label="question",show_label=False,placeholder="Enter text and press enter, or press the send button.",scale=20, max_lines=3, lines=3)
                    send_btn = gr.Button("Send",elem_id="send-btn",scale=1,size="sm")
        
            # 右侧面板 
            with gr.Column(scale=5):
                # 右上图片显示 
                image_box = gr.HTML(label="result",show_label=False,elem_id="right-column")
        # with gr.Tab("Database",elem_id="database-tab"):
        #     with gr.Row():
        #         with gr.Column(scale=1):
        #             gr.Markdown("# Database")
        #             with gr.Accordion("Entities choose", open=True): 
        #                 # 全选控件 
        #                 select_all = gr.Checkbox(label="Select All",value=True)
        #                 # 多选控件组 
        #                 checkbox_group = gr.CheckboxGroup(value=choices,
        #                     show_label=False,
        #                     choices=choices,
        #                     interactive=True 
        #                 )
        #         with gr.Column(scale=4):
        #             df = gr.Dataframe()
            
    
        # 事件绑定
        # send_btn.click(fn=user_msg,
        #                 inputs=[msg,chatbot],
        #                 outputs=[msg,chatbot]
        #                 ).then(
        #                 fn=respond,
        #                 inputs=[chatbot],
        #                 outputs=[chatbot, image_box]
        #                 )
        
        # # 回车提交支持 
        # msg.submit(fn=user_msg,
        #                 inputs=[msg,chatbot],
        #                 outputs=[msg,chatbot]
        #                 ).then(
        #                 fn=respond,
        #                 inputs=[chatbot],
        #                 outputs=[chatbot, image_box]
        #                 )
        send_btn.click(fn=yield_respond,
                    inputs=[msg, chatbot],
                    outputs=[msg, chatbot, image_box]
                    )
        msg.submit(fn=yield_respond,
                    inputs=[msg, chatbot],
                    outputs=[msg, chatbot, image_box]
                    )

        # 重置对话
        # 显示弹窗逻辑
        new_chat.click(
            lambda: gr.update(visible=True),  # 显示弹窗
            outputs=confirm_dialog
        )
        
        # 确认操作
        confirm_btn.click(
            lambda: [INITIAL_CHATBOT, INITIAL_HTML, INITIAL_MESSAGE_BOX, gr.update(visible=False)],
            outputs=[chatbot, image_box, msg, confirm_dialog]
        )
        
        # 取消操作
        cancel_btn.click(
            lambda: gr.update(visible=False),
            outputs=confirm_dialog
        )
        set_parameters.click(
            # fn=api_check,
            # inputs=[model_selector, api_key_input],
            lambda: gr.update(visible=True),
            outputs=confirm_set_dialog
        )
        
        # 确认操作
        confirm_set_btn.click(lambda :[INITIAL_CHATBOT, INITIAL_HTML, INITIAL_MESSAGE_BOX, gr.update(visible=False), gr.update(visible=True)],
            inputs=[],
            outputs=[chatbot, image_box, msg, confirm_set_dialog, waiting_text]
        ).then(
            fn=interaction_false,
            outputs=[new_chat, send_btn, set_parameters, model_selector, top_k, hops]
        ).then(
            fn=parameters_embedding_live_update,
            inputs=[model_selector,api_key_input, top_k, hops],
            outputs=[],
        ).then(
            lambda: [gr.update(visible=False), gr.update(open=False)],
            outputs=[waiting_text, model_parameter]
        ).then(
            fn=update_state,
            inputs=[model_selector, top_k, hops],
            outputs=[current_model_selector, current_top_k, current_hops] # 返回更新后的状态
        ).then(
            fn=interaction_true,
            outputs=[new_chat, send_btn, set_parameters, model_selector, top_k, hops]
        )
        
        # 取消操作
        cancel_set_btn.click(
            lambda: gr.update(visible=False),
            outputs=confirm_set_dialog
        )
        file_box.change( 
            fn=process_file,
            inputs=[file_box, gr.State(file_dropdown.choices)],
            outputs=[file_dropdown],
            api_name="process"
        ).then(
            lambda: None,
            outputs=file_box
        )

        select_base.change(
            select_base_change,
            inputs=[select_base],
            outputs=[file_dropdown]
        )
        demo.load(fn=None,  js="""
            () => {
                document.getElementById("send-btn").disabled  = true;
                return [];
            }
            """)
        msg.change( 
            fn=None,
            js="""
            (text) => {
                const btn = document.getElementById("send-btn"); 
                btn.disabled  = (text.trim()  === "");
                return [];
            }
            """,
            inputs=[msg],
            outputs=[]
        )
        demo.load(fn=None,  js="""
            () => {
                document.getElementById("parameter-btn").disabled  = true;
                return [];
            }
            """)
        model_selector.change(
            fn=None, # 当值改变时调用这个函数
            inputs=[model_selector, top_k, hops, current_model_selector, current_top_k, current_hops],
            outputs=[],
            js="""
            (model_selector, top_k, hops, current_model_selector, current_top_k, current_hops) => {
                const btn = document.getElementById("parameter-btn");
                btn.disabled  = (model_selector === current_model_selector && top_k === current_top_k && hops === current_hops);
                return [];
            }""")
        top_k.change(
            fn=None, # 当值改变时调用这个函数
            inputs=[model_selector, top_k, hops, current_model_selector, current_top_k, current_hops],
            outputs=[],
            js="""
            (model_selector, top_k, hops, current_model_selector, current_top_k, current_hops) => {
                const btn = document.getElementById("parameter-btn");
                btn.disabled  = (model_selector === current_model_selector && top_k === current_top_k && hops === current_hops);
                return [];
            }""")
        hops.change(
            fn=None, # 当值改变时调用这个函数
            inputs=[model_selector, top_k, hops, current_model_selector, current_top_k, current_hops],
            outputs=[],
            js="""
            (model_selector, top_k, hops, current_model_selector, current_top_k, current_hops) => {
                const btn = document.getElementById("parameter-btn");
                btn.disabled  = (model_selector === current_model_selector && top_k === current_top_k && hops === current_hops);
                return [];
            }""")
        # model_selector.change(
        #     fn=update_api_key_visibility, # 当值改变时调用这个函数
        #     inputs=model_selector,        # 将 model_selector 的当前值作为输入传给函数
        #     outputs=api_key_input         # 函数的返回值（gr.update对象）将作用于 api_key_input 组件
        # )


    # select_all.change( 
    #     fn=toggle_select_all,
    #     inputs=[select_all, checkbox_group],
    #     outputs=checkbox_group 
    # )
    # checkbox_group.change(
    #     fn=toggle_select_all_reverse,
    #     inputs=checkbox_group,
    #     outputs=select_all
    # )
    return demo

if __name__ == "__main__":
    demo = ui()
    demo.launch(show_api=False, show_error=False, ssl_verify=False, analytics_enabled=False, share=False) 
