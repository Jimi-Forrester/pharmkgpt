from src.ui.utils import *
from src.ui.theme import Kotaemon

import gradio as gr
from src.ui.config import *
import os
from PIL import Image
os.environ["GRADIO_ANALYTICS"] = "False"

def ui():
    base_dir = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
    with open(f"{base_dir}/gradio.css", "r", encoding="utf-8") as f:
        custom_css = f.read()
    logo_img = Image.open(f"{base_dir}/static/logo.png")  # 确保路径正确
    icon = """<link rel="icon" type="image/png" href="/file=src/ui/static/PharmKGPT.png">"""
    with gr.Blocks(theme=Kotaemon(text_size="lg").set(body_background_fill='white',background_fill_primary='white', section_header_text_weight=700),title="PharmKGPT", css=custom_css, fill_width=True, head=icon) as demo:
    # with gr.Tab("KG2RAG",elem_id="chat-tab"):
        with gr.Row():
            # 左侧设置面板 
            with gr.Column(scale=1,elem_id="left-column", min_width=300) as sidebar_column:
                # gr.Markdown("# PharmKGPT",height="15vh")
                gr.Image(value=logo_img, height=100, show_label=False, show_download_button=False, show_fullscreen_button=False, elem_id="logo-img-container")
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
                with gr.Accordion(label="Run settings", elem_id="run-settings") as model_parameter:
                    current_model_selector, current_top_k, current_hops = gr.State(None), gr.State(None), gr.State(None)
                    query_result = gr.State(None)
                    is_parameter_set = gr.State(False)
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
                    top_k = gr.Slider(minimum=1, maximum=15, value=5, step=1, label="Max Retrieved",interactive=True, elem_id="top-k")
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
                            gr.Markdown("### Confirm to set parameters?",)
                        confirm_set_btn = gr.Button("Confirm", variant="stop")
                        cancel_set_btn = gr.Button("Cancel")
                    waiting_text = gr.Markdown(
                        "### Waiting for parameters to be set...",
                        visible=False, 
                    )
                with gr.Accordion(label="Entity statistics", open=True, elem_id="entity-counter") as entity_counter:
                    with gr.Row():
                        plot = gr.Plot(show_label=False)
                with gr.Accordion(label="Highlight", open=True, elem_id="highlight"):
                    gr.Markdown("""                      
                        *  **Accurate Answering**: Reliable, source-backed responses specifically addressing Delirium & AD inquiries.
                        *  **Explore Complex Relationships**: Visualize links across disease concepts, proteomics, metabolomics, genetics and pathways using our knowledge graph.
                        *  **Efficient Information Synthesis**: Rapidly extract key insights and data from extensive research literature.
                        """
                    )
                with gr.Accordion(label="Announcement", open=True, elem_id="announcement"):
                    gr.Markdown(
                            """
                    *  **Knowledge Base Update**: Delirium & Alzheimer's literature updated through March 2025.
                    *  **Example**: How does kynurenic acid contribute to delirium?
                    *  **Tip**: Search results might be better when *Max Retrieved* is set to 5 or less.
                    *  **Feedback Welcome**: Please share your thoughts or report any issues to xianglu@mindrank.ai.
                    """  
                        )
                    
                # with gr.Accordion(label="Knowledge base", visible=False) as knowledge_base:
                #     with gr.Row():
                #         select_base = gr.CheckboxGroup(choices=["PubMed", "File(s)"], value=["PubMed"], interactive=True, show_label=False)
                #         file_dropdown = gr.Dropdown(
                #             choices=['file1', 'file2', 'file3'], multiselect=True, interactive=True, show_label=False, visible=False)
                # with gr.Accordion(label="File upload", open=False, visible=False) as file_input_section:
                #     with gr.Row():
                #         file_box = gr.File(file_count="single", file_types=['pdf','txt'], type="filepath", interactive=True, show_label=False, height=140)
                #         progress_bar = gr.Textbox(label="处理状态", visible=False)


            with gr.Column(scale=4,elem_id="mid-column"):
                sidebar_visible = gr.State(True)
                toggle_button = gr.Button("〈 Hide", size="sm", scale=0, elem_id="toggle-button")
                chatbot = gr.Chatbot(value=[],
                                    height="85vh",label="Chat",
                                    placeholder="<p style='font-size:18px; font-weight:bold'>You can ask questions.</p>",
                                    show_label=False,
                                    elem_id="main-chat-bot",
                                    show_copy_button=True,
                                    likeable=False,
                                    bubble_full_width=False)
                with gr.Row(equal_height=True):
                    msg = gr.Textbox(label="question",show_label=False, placeholder="Input message and press shift + enter, or press the send button.",scale=20, max_lines=3, lines=3)
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
        send_btn.click(fn=get_parameter_in_send,
                        inputs=[],
                        outputs=[]
                        ).success(fn=user_msg,
                        inputs=[msg,chatbot],
                        outputs=[msg,chatbot]
                        ).success(
                        fn=yield_respond,
                        inputs=[chatbot],
                        outputs=[chatbot, query_result]
                        ).success(
                        fn=answer_respond,
                        inputs=[chatbot, query_result],
                        outputs=[chatbot, image_box]
                        )
        
        # # 回车提交支持 
        msg.submit(fn=get_parameter_in_send,
                        inputs=[],
                        outputs=[]
                        ).success(fn=user_msg,
                        inputs=[msg,chatbot],
                        outputs=[msg,chatbot]
                        ).success(
                        fn=yield_respond,
                        inputs=[chatbot],
                        outputs=[chatbot, query_result]
                        ).success(
                        fn=answer_respond,
                        inputs=[chatbot, query_result],
                        outputs=[chatbot, image_box]
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
            fn=api_check,
            inputs=[model_selector, api_key_input],
            # lambda: gr.update(visible=True),
            outputs=confirm_set_dialog
        )
        
        # 确认操作
        confirm_set_btn.click(lambda :[INITIAL_CHATBOT, INITIAL_HTML, INITIAL_MESSAGE_BOX, gr.update(visible=False), gr.update(visible=True)],
            inputs=[],
            outputs=[chatbot, image_box, msg, confirm_set_dialog, waiting_text]
        ).then(
            fn=interaction_false,
            inputs=[new_chat, msg, send_btn, set_parameters, model_selector, top_k, hops],
            outputs=[new_chat, msg, send_btn, set_parameters, model_selector, top_k, hops]
        ).then(
            fn=parameters_embedding_live_update,
            inputs=[model_selector,api_key_input, top_k, hops],
            outputs=[],
        ).then(
            lambda: [gr.update(visible=False), gr.update(open=False), gr.update(value=True), gr.update(placeholder="<p style='font-size:18px; font-weight:bold'>You can ask questions.</p>")],
            outputs=[waiting_text, model_parameter, is_parameter_set, chatbot]
        ).then(
            fn=update_state,
            inputs=[model_selector, top_k, hops],
            outputs=[current_model_selector, current_top_k, current_hops] # 返回更新后的状态
        ).then(
            fn=interaction_true,
            inputs=[new_chat, msg, send_btn, set_parameters, model_selector, top_k, hops],
            outputs=[new_chat, msg, send_btn, set_parameters, model_selector, top_k, hops]
        )
        
        # 取消操作
        cancel_set_btn.click(
            lambda: gr.update(visible=False),
            outputs=confirm_set_dialog
        )
        demo.load(fn=update_placeholder_in_load,
                    inputs=[],
                    outputs=[chatbot, is_parameter_set, model_selector, top_k]
            )
        demo.load(fn=None,  js="""
            () => {
                document.getElementById("send-btn").disabled  = true;
                return [];
            }
            """)
        demo.load(fn=plot_interactive_hbar,  outputs=plot)

        msg.change( 
            fn=None,
            js="""
            (text, is_parameter_set) => {
                const btn = document.getElementById("send-btn"); 
                btn.disabled = (text.trim() === "");
                return [];
            }
            """,
            inputs=[msg, is_parameter_set],
            outputs=[]
        )
        toggle_button.click(
                fn=toggle_sidebar,         # The function to call
                inputs=[sidebar_visible],  # Pass the current state to the function
                outputs=[
                    sidebar_column,        # The component to update visibility for
                    toggle_button,         # The button component to update its text/icon
                    sidebar_visible        # The state variable to update
                ]
            )
        model_selector.change(
            fn=update_api_key_visibility, # 当值改变时调用这个函数
            inputs=model_selector,        # 将 model_selector 的当前值作为输入传给函数
            outputs=api_key_input         # 函数的返回值（gr.update对象）将作用于 api_key_input 组件
        )

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
