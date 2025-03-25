from src.ui.main import ui
import os
os.environ["GRADIO_ANALYTICS"] = "False"

demo = ui()

if __name__ == "__main__":
    demo.launch(show_api=False, show_error=False)