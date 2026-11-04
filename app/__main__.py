"""Run the web app:  python -m app [--port 5050]"""
import argparse
import logging

from .server import create_app

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5050)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    create_app().run(host=args.host, port=args.port, debug=False, threaded=False)
