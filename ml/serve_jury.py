"""Standalone API without training videos or annotation files.

Run from the extracted runtime directory: .venv/bin/python ml/serve_jury.py
The key is read from .secrets/jury_api_key, with file permissions 0600.
"""
import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from jury_api import JuryAPI


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    api = JuryAPI(root)

    class Handler(BaseHTTPRequestHandler):
        def json_response(self, value, status=200):
            data = json.dumps(value, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if not api.handle_get(self):
                self.json_response({'error': 'Not found'}, 404)

        def do_POST(self):
            if not api.handle_post(self):
                self.json_response({'error': 'Not found'}, 404)

    ThreadingHTTPServer(('127.0.0.1', args.port), Handler).serve_forever()


if __name__ == '__main__':
    main()
