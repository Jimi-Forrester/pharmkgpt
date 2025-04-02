from src.ui.main import ui
import os
os.environ["GRADIO_ANALYTICS"] = "False"

demo = ui()

if __name__ == "__main__":
    demo.launch(ssl_verify=False, quiet=True, share=False)