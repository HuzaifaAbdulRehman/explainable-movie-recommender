import csv
import json
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from urllib.parse import parse_qs, urlsplit

from recommender import Model

ROOT = Path(__file__).resolve().parent
LOCK = Lock()
METHODS = {"Top-k", "MMR", "Genre quota", "QUBO", "Tie"}


class Handler(BaseHTTPRequestHandler):
    def send(self, body, content_type="application/json", status=200):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def json(self, value, status=200):
        self.send(json.dumps(value).encode("utf-8"), status=status)

    def do_GET(self):
        url = urlsplit(self.path)
        if url.path == "/":
            self.send((ROOT / "index.html").read_bytes(), "text/html; charset=utf-8")
        elif url.path == "/api/users":
            self.json(list(self.server.model.history))
        elif url.path == "/api/movies":
            movies = self.server.model.movies
            self.json([{"id": item, "title": title} for item, (title, _) in movies.items()])
        elif url.path == "/api/results":
            path = ROOT / "results.json"
            self.json(json.loads(path.read_text(encoding="utf-8")) if path.exists() else {})
        elif url.path == "/api/recommend":
            fields = parse_qs(url.query)
            model = self.server.model
            try:
                if "movie" in fields:
                    history = list(dict.fromkeys(map(int, fields["movie"])))
                    if not 3 <= len(history) <= 20 or any(item not in model.movies for item in history):
                        self.json({"error": "Choose 3 to 20 distinct movies from the catalog."}, 400)
                        return
                else:
                    history = model.history[int(fields["user"][0])]
                diversity = float(fields["diversity"][0])
                balance = float(fields["balance"][0])
                if not (0 <= diversity <= 1 and 0 <= balance <= 1):
                    raise ValueError
            except (ValueError, KeyError):
                self.json({"error": "Choose a valid user and weights between 0 and 1."}, 400)
                return
            self.json({"history": [model.movies[item][0] for item in history],
                       "methods": model.run_history(history, diversity, balance)})
        else:
            self.json({"error": "Page not found."}, 404)

    def do_POST(self):
        if self.path != "/api/judgment":
            self.json({"error": "Page not found."}, 404)
            return
        try:
            size = int(self.headers["Content-Length"])
            if not 0 < size <= 4096:
                raise ValueError
            data = json.loads(self.rfile.read(size))
            user = int(data["user"])
            diversity, balance = float(data["diversity"]), float(data["balance"])
            relevance, variety = data["relevance"], data["variety"]
            if (user not in self.server.model.history or
                    not (0 <= diversity <= 1 and 0 <= balance <= 1) or
                    relevance not in METHODS or variety not in METHODS):
                raise ValueError
        except (ValueError, KeyError, TypeError):
            self.json({"error": "Select a method or tie for both judgments."}, 400)
            return

        path = ROOT / "judgments.csv"
        with LOCK, path.open("a+", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            if stream.tell() == 0:
                writer.writerow(["timestamp_utc", "user", "diversity", "balance",
                                 "relevance_preference", "variety_preference"])
            writer.writerow([datetime.now(timezone.utc).isoformat(), user, diversity,
                             balance, relevance, variety])
        self.json({"saved": True}, 201)


if __name__ == "__main__":
    model = Model()
    with ThreadingHTTPServer(("127.0.0.1", 8765), Handler) as server:
        server.model = model
        print("Open http://127.0.0.1:8765", flush=True)
        server.serve_forever()
