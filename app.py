"""Local web interface for the predefined-URL mini search engine."""

from __future__ import annotations

import html
import os
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Lock
from urllib.parse import parse_qs, urlsplit

from search_engine import SearchEngine


HOST = "127.0.0.1"
PORT = int(os.environ.get("PORT", "8000"))
MAX_FORM_BYTES = 64 * 1024

_engine: SearchEngine | None = None
_engine_urls: tuple[str, ...] = ()
_engine_lock = Lock()


def _escape(value: str) -> str:
    return html.escape(value, quote=True)


def _safe_snippet(snippet: str) -> str:
    """Escape scraped content while retaining only the engine's mark tags."""
    escaped = _escape(snippet)
    escaped = re.sub(r"&lt;mark&gt;", "<mark>", escaped, flags=re.IGNORECASE)
    escaped = re.sub(r"&lt;/mark&gt;", "</mark>", escaped, flags=re.IGNORECASE)
    return escaped


def _page(
    *,
    urls: str = "",
    query: str = "",
    ranking: str = "hybrid",
    results: list[dict] | None = None,
    error: str = "",
) -> bytes:
    ranking_options = "".join(
        f'<option value="{value}"{" selected" if ranking == value else ""}>{label}</option>'
        for value, label in (
            ("hybrid", "Hybrid: relevance + PageRank"),
            ("old", "Old way: word count"),
            ("surfer", "Surfer way: random surfer"),
        )
    )
    result_html = ""
    if results is not None:
        if results:
            cards = []
            for result in results:
                score = _escape(result["score_display"])
                cards.append(
                    '<article class="result">'
                    f'<h2><a href="{_escape(result["url"])}" target="_blank" '
                    f'rel="noopener noreferrer">{_escape(result["title"])}</a></h2>'
                    f'<p class="url">{_escape(result["url"])}</p>'
                    f'<p class="snippet">{_safe_snippet(result["snippet"])}</p>'
                    f'<p class="score">Score: {score}</p>'
                    "</article>"
                )
            result_html = (
                f'<section class="results"><h2>Results ({len(results)})</h2>'
                + "".join(cards)
                + "</section>"
            )
        else:
            result_html = '<p class="notice">No matching pages found.</p>'

    error_html = f'<p class="error">{_escape(error)}</p>' if error else ""
    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Mini Search</title>
  <style>
    :root {{ color-scheme: light; font-family: system-ui, sans-serif; color: #172033; background: #f5f7fb; }}
    body {{ margin: 0; }}
    main {{ max-width: 820px; margin: 48px auto; padding: 0 20px; }}
    h1 {{ margin-bottom: 8px; }}
    .intro, .help {{ color: #586174; }}
    form, .result {{ background: white; border: 1px solid #dfe4ee; border-radius: 12px; padding: 20px; }}
    label {{ display: block; font-weight: 650; margin: 16px 0 7px; }}
    textarea, input, select {{ box-sizing: border-box; width: 100%; padding: 11px 12px; border: 1px solid #c8cfdd; border-radius: 7px; font: inherit; }}
    textarea {{ min-height: 120px; resize: vertical; }}
    button {{ margin-top: 20px; padding: 11px 18px; border: 0; border-radius: 7px; background: #2257d6; color: white; font: inherit; font-weight: 650; cursor: pointer; }}
    button:hover {{ background: #1948b8; }}
    .results {{ margin-top: 28px; }}
    .result {{ margin: 12px 0; }}
    .result h2 {{ margin: 0; font-size: 1.15rem; }}
    a {{ color: #2257d6; }}
    .url, .score, .help {{ font-size: .9rem; }}
    .url {{ overflow-wrap: anywhere; color: #667085; }}
    .snippet {{ line-height: 1.6; }}
    mark {{ background: #fff0a8; }}
    .error {{ color: #a12622; background: #fff0ef; border-radius: 8px; padding: 12px; }}
    .notice {{ color: #586174; }}
    @media (max-width: 600px) {{ main {{ margin: 28px auto; }} }}
  </style>
</head>
<body>
  <main>
    <h1>Mini Search</h1>
    <p class="intro">Search pages from a list of URLs you provide.</p>
    {error_html}
    <form method="post">
      <label for="urls">Target URLs</label>
      <textarea id="urls" name="urls" required placeholder="https://example.com/&#10;https://example.org/">{_escape(urls)}</textarea>
      <p class="help">Enter one http(s) URL per line. Indexing and the one-time surfer simulation run when you submit a new URL list.</p>
      <label for="query">Search query</label>
      <input id="query" name="query" value="{_escape(query)}" placeholder="What are you looking for?" required>
      <label for="ranking">Ranking method</label>
      <select id="ranking" name="ranking">{ranking_options}</select>
      <button type="submit">Search</button>
    </form>
    {result_html}
  </main>
</body>
</html>"""
    return document.encode("utf-8")


class SearchHandler(BaseHTTPRequestHandler):
    """Handle the search form on a local-only HTTP server."""

    def do_GET(self) -> None:
        self._send_html(_page())

    def do_POST(self) -> None:
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send_html(_page(error="Invalid form submission."), status=400)
            return
        if content_length <= 0 or content_length > MAX_FORM_BYTES:
            self._send_html(_page(error="Form submission is empty or too large."), status=413)
            return

        form_data = parse_qs(self.rfile.read(content_length).decode("utf-8", errors="replace"))
        urls_text = form_data.get("urls", [""])[0]
        query = form_data.get("query", [""])[0].strip()
        ranking = form_data.get("ranking", ["hybrid"])[0]
        urls = tuple(dict.fromkeys(line.strip() for line in urls_text.splitlines() if line.strip()))

        if not urls:
            self._send_html(_page(query=query, ranking=ranking, error="Enter at least one URL."))
            return
        if ranking not in {"hybrid", "old", "surfer"}:
            self._send_html(_page(urls=urls_text, query=query, error="Choose a valid ranking method."))
            return
        if any(urlsplit(url).scheme not in {"http", "https"} or not urlsplit(url).netloc for url in urls):
            self._send_html(
                _page(urls=urls_text, query=query, ranking=ranking,
                      error="Every target must be an absolute http:// or https:// URL.")
            )
            return
        if not query:
            self._send_html(
                _page(urls=urls_text, ranking=ranking, error="Enter a search query.")
            )
            return

        try:
            engine = self._get_engine(urls)
            results = engine.search(query, ranking=ranking)
        except (RuntimeError, ValueError) as exc:
            self._send_html(
                _page(urls=urls_text, query=query, ranking=ranking, error=str(exc)),
                status=502,
            )
            return

        self._send_html(
            _page(urls=urls_text, query=query, ranking=ranking, results=results)
        )

    def _get_engine(self, urls: tuple[str, ...]) -> SearchEngine:
        global _engine, _engine_urls
        with _engine_lock:
            if _engine is None or urls != _engine_urls:
                _engine = SearchEngine(urls)
                _engine_urls = urls
            return _engine

    def _send_html(self, body: bytes, *, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        print(f"{self.address_string()} - {format % args}")


if __name__ == "__main__":
    server = ThreadingHTTPServer((HOST, PORT), SearchHandler)
    print(f"Mini Search is running at http://{HOST}:{PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down Mini Search.")
        server.server_close()
