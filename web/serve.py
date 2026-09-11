"""
Static file server for the JS dashboard, with HTTP Range support.

Python's stdlib SimpleHTTPRequestHandler does NOT implement Range requests, and browsers
require them to seek within a <video> (without it, scrubbing the timeline silently fails or
the video refuses to play at all). This adds the ~40 lines of Range handling needed, rather
than pulling in a web framework just to serve four static files.

Usage: python web/serve.py [port]   (default 8600)
Then open http://localhost:8600
"""
import os
import re
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

WEB_DIR = os.path.dirname(os.path.abspath(__file__))
RANGE_RE = re.compile(r"bytes=(\d*)-(\d*)")


class RangeRequestHandler(SimpleHTTPRequestHandler):
    # SimpleHTTPRequestHandler defaults to HTTP/1.0, which closes the connection after every
    # response and advertises no keep-alive. curl does not care - it pulled the 63 MB clip in
    # 0.49 s - but Chrome's media pipeline does: on an HTTP/1.0 response it never progressed
    # past readyState 0 for the 90 s clips, so the video panel stayed black while the overlay
    # canvas drew normally. HTTP/1.1 is safe here because every response this handler sends
    # carries an accurate Content-Length (the Range branch sets it explicitly, and the base
    # class sets it for whole files).
    protocol_version = "HTTP/1.1"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=WEB_DIR, **kwargs)

    def end_headers(self):
        # Range support has to be advertised or some browsers won't even try to seek.
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def send_head(self):
        range_header = self.headers.get("Range")
        if not range_header:
            return super().send_head()

        path = self.translate_path(self.path)
        if not os.path.isfile(path):
            return super().send_head()

        match = RANGE_RE.match(range_header)
        if not match:
            return super().send_head()

        file_size = os.path.getsize(path)
        start_raw, end_raw = match.groups()
        if start_raw:
            start = int(start_raw)
            end = int(end_raw) if end_raw else file_size - 1
        else:
            # Suffix form ("bytes=-500") means the LAST N bytes, not from 0 - browsers use
            # this to read an MP4's trailing moov atom before deciding how to stream.
            start = max(0, file_size - int(end_raw))
            end = file_size - 1
        end = min(end, file_size - 1)
        if start > end:
            self.send_error(416, "Requested range not satisfiable")
            return None

        f = open(path, "rb")
        f.seek(start)
        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        # copyfile() would send the rest of the file; cap it to the requested slice.
        self._range_remaining = end - start + 1
        return _LimitedReader(f, self._range_remaining)

    def log_message(self, fmt, *args):
        if "404" in (fmt % args):  # keep genuine misses visible, drop the 200-per-asset noise
            super().log_message(fmt, *args)


class _LimitedReader:
    """Wraps a file object so copyfile() stops at the end of the requested byte range."""

    def __init__(self, fileobj, limit):
        self._f = fileobj
        self._remaining = limit

    def read(self, size=-1):
        if self._remaining <= 0:
            return b""
        if size < 0 or size > self._remaining:
            size = self._remaining
        data = self._f.read(size)
        self._remaining -= len(data)
        return data

    def close(self):
        self._f.close()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8600
    if not os.path.exists(os.path.join(WEB_DIR, "data", "index.json")):
        raise SystemExit("No exported data found - run `python web/export_data.py` first.")
    print(f"Drone analytics dashboard: http://localhost:{port}")
    print("Ctrl+C to stop.")
    # ThreadingHTTPServer, NOT HTTPServer: a playing <video> issues a continuous stream of
    # Range requests, and on a single-threaded server every one of them queues behind whatever
    # else is in flight - a 6 MB data fetch, a burst of map tiles. That starves the video's
    # buffer and shows up as playback stalling, which is easy to misread as "the video is too
    # big". Threading lets media requests be served while other work is still going.
    srv = ThreadingHTTPServer(("127.0.0.1", port), RangeRequestHandler)
    srv.daemon_threads = True     # don't let in-flight requests block Ctrl+C
    srv.serve_forever()
