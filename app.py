"""HF Space entrypoint. Reuses the Gradio demo defined in demo/app.py."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from demo.app import build_demo  # noqa: E402

if __name__ == "__main__":
    demo = build_demo()
    demo.launch()
