"""
Web UI entry point.
"""
import argparse

from ai_workbench.config import get_config
from ai_workbench.web.app import launch_app


def main() -> None:
    config = get_config()
    parser = argparse.ArgumentParser(description="Launch AI Workbench Web UI")
    parser.add_argument("--host", default=config.web_host, help="Host to bind (default from config)")
    parser.add_argument("--port", type=int, default=config.web_port, help="Port to bind (default from config)")
    parser.add_argument("--share", action="store_true", default=config.web_share, help="Enable public sharing")
    args = parser.parse_args()

    launch_app(host=args.host, port=args.port, share=args.share)


if __name__ == "__main__":
    main()
