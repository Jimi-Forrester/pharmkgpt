from src.ui.utils import respond,user_msg,toggle_select_all,toggle_select_all_reverse, create_app, parameters_embedding, interaction_true, interaction_false
from src.ui.theme import Kotaemon

import gradio as gr
from src.ui.config import *
import os

os.environ["GRADIO_ANALYTICS"] = "False"


def ui():
    base_dir = os.path.dirname(os.path.abspath(__file__)) 
    with open(f"{base_dir}/gradio.css", "r", encoding="utf-8") as f:
        custom_css = f.read()

    with gr.Blocks(theme=Kotaemon(),title="KG2RAG",css=custom_css,fill_width=True) as demo:
    # with gr.Tab("KG2RAG",elem_id="chat-tab"):
        with gr.Row():
            # 左侧设置面板 
            with gr.Column(scale=1,min_width=300,elem_id="left-column"):
                gr.Markdown("# KG2RAG",height="10vh")
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
                with gr.Accordion(label="Model paremeter") as model_parameter:
                    model_selector = gr.Dropdown(
                                    label="Model",
                                    choices=MODELS,
                                    value=MODELS[0],  # 默认选中第一个
                                    interactive=True,
                                )
                    hops = gr.Slider(minimum=1, maximum=5, value=1, step=1, label="Multi-hop expansion",interactive=True)
                    top_k = gr.Slider(minimum=1, maximum=20, value=5, step=1, label="Top N retrieval",interactive=True)
                    set_parameters = gr.Button(
                        value="Set parameters",
                        min_width=120,
                        size="sm",
                        variant="primary",
                        elem_id="styled-btn",
                    )
                    with gr.Row(visible=False) as confirm_set_dialog:
                        with gr.Row():
                            gr.Markdown("### Confirm to set model parameters?",)
                        confirm_set_btn = gr.Button("Confirm", variant="stop")
                        cancel_set_btn = gr.Button("Cancel")
                    waiting_text = gr.Markdown(
                        label="### Waiting for model parameters to be set...",
                        visible=False, 
                    )
            with gr.Column(scale=4):
                chatbot = gr.Chatbot(value=[],
                                    height="80vh",label="Chat",
                                    placeholder="You can ask questions.",
                                    show_label=False,
                                    elem_id="main-chat-bot",
                                    show_copy_button=True,
                                    likeable=False,
                                    bubble_full_width=False,)
                with gr.Row(equal_height=True):
                    msg = gr.Textbox(label="question",show_label=False,placeholder="Enter text and press enter, or press the send button.",scale=20)
                    submit_btn = gr.Button("Send",elem_id="styled-btn",scale=1,size="sm")
        
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
        submit_btn.click(fn=user_msg,
                        inputs=[msg,chatbot],
                        outputs=[msg,chatbot]
                        ).then(
                        fn=respond,
                        inputs=[chatbot],
                        outputs=[chatbot, image_box]
                        )
        
        # 回车提交支持 
        msg.submit(fn=user_msg,
                        inputs=[msg,chatbot],
                        outputs=[msg,chatbot]
                        ).then(
                        fn=respond,
                        inputs=[chatbot],
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
            lambda: gr.update(visible=True),  # 显示弹窗
            outputs=confirm_set_dialog
        )
        
        # 确认操作
        confirm_set_btn.click(lambda :[INITIAL_CHATBOT, INITIAL_HTML, INITIAL_MESSAGE_BOX, gr.update(visible=False),gr.update(visible=True)],
            inputs=[],
            outputs=[chatbot, image_box, msg, confirm_set_dialog, waiting_text]
        ).then(
            fn=interaction_false,
            outputs=[new_chat, submit_btn, set_parameters, model_selector, top_k, hops]
        ).then(
            fn=parameters_embedding,
            inputs=[model_selector,top_k,hops],
            outputs=[waiting_text],
        ).then(
            fn=interaction_true,
            outputs=[new_chat, submit_btn, set_parameters, model_selector, top_k, hops]
        )
        
        # 取消操作
        cancel_set_btn.click(
            lambda: gr.update(visible=False),
            outputs=confirm_set_dialog
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
    demo.launch(show_api=False, show_error=False,) 
