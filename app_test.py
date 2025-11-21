import os
import pytest

if os.getenv("RUN_HEAVY_RAG_TESTS", "0") != "1":
    pytest.skip(
        "Skipping FastAPI startup smoke test because RUN_HEAVY_RAG_TESTS is disabled.",
        allow_module_level=True,
    )

import uvicorn

from src.main import app


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5001)
