import os
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
from src.ui.main import ui
import mimetypes
mimetypes.add_type("image/svg+xml", ".svg")

demo = ui()

if __name__ == "__main__":
    demo.launch(ssl_verify=False, share=False, allowed_paths=["src/ui/static"])