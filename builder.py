import os
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
from src.ui.main import ui

demo = ui()

if __name__ == "__main__":
    demo.launch(ssl_verify=False, share=False, allowed_paths=["src/ui"])