"""Same-origin bridge to the local Next.js UI; no arbitrary upstream URLs."""
import http.client
import os
from urllib.parse import urlsplit

HOP_HEADERS = {'connection', 'keep-alive', 'proxy-authenticate', 'proxy-authorization',
               'te', 'trailer', 'transfer-encoding', 'upgrade'}


def frontend_path(path):
    return path in {'/', '/analysis', '/favicon.ico', '/icon.svg', '/api/analysis'} or path.startswith(
        ('/analysis/', '/_next/', '/api/jobs/', '/api/media/', '/demo/'))


def forward(handler, method):
    """Return False when frontend forwarding is disabled or route is not owned."""
    port = os.environ.get('NUHACK_FRONTEND_PORT')
    path = urlsplit(handler.path).path
    if not port or not frontend_path(path):
        return False
    if method not in ('GET', 'HEAD', 'POST') or (method == 'POST' and path != '/api/analysis'):
        handler.json_response({'error': 'Method not allowed'}, 405)
        return True
    conn = None
    responded = False
    try:
        length = 0
        if method == 'POST':
            if handler.headers.get('Transfer-Encoding') or not handler.headers.get('Content-Length'):
                handler.json_response({'error': 'Content-Length required'}, 411)
                return True
            length = int(handler.headers['Content-Length'])
            if not 0 < length <= 64 * 1024 * 1024 + 1024 * 1024:
                handler.json_response({'error': 'Upload exceeds 64 MiB plus form headers'}, 413)
                return True
        conn = http.client.HTTPConnection('127.0.0.1', int(port), timeout=125)
        conn.putrequest(method, handler.path, skip_host=True, skip_accept_encoding=True)
        for key, value in handler.headers.items():
            if key.lower() not in HOP_HEADERS | {'x-api-key'}:
                conn.putheader(key, value)
        conn.putheader('Connection', 'close')
        conn.endheaders()
        handler.connection.settimeout(125)
        remaining = length
        while remaining:
            part = handler.rfile.read(min(256 * 1024, remaining))
            if not part:
                raise ConnectionError('Upload interrupted')
            conn.send(part)
            remaining -= len(part)
        response = conn.getresponse()
        handler.send_response(response.status)
        for key, value in response.getheaders():
            if key.lower() not in HOP_HEADERS:
                handler.send_header(key, value)
        handler.send_header('Connection', 'close')
        handler.end_headers()
        responded = True
        handler.close_connection = True
        if method != 'HEAD':
            while part := response.read(256 * 1024):
                handler.wfile.write(part)
    except (OSError, ValueError, http.client.HTTPException):
        if not responded:
            handler.json_response({'error': 'Интерфейс временно недоступен; повторите запрос.'}, 502)
    finally:
        if conn:
            conn.close()
    return True
