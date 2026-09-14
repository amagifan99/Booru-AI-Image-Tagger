import http.server
import socketserver
import webbrowser
import threading
import os

PORT = 8000
DIRECTORY = os.path.abspath(".")  # Change to your folder path

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

def start_server():
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"Serving at http://localhost:{PORT}")
        httpd.serve_forever()

if __name__ == "__main__":
    # Start server in a separate thread so it doesn't block
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()

    # Open the viewer in default browser
    webbrowser.open(f"http://localhost:{PORT}/viewer.html")

    # Keep the main thread alive while server runs
    try:
        while True:
            pass
    except KeyboardInterrupt:
        print("\nServer stopped.")
