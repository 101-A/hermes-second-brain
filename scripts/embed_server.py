#!/usr/bin/env python3
"""Local embeddings + rerank service: OpenAI-shaped /v1/embeddings, Cohere-shaped /v1/rerank.

    python3 scripts/embed_server.py
    EMBED_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 PORT=8099 \
        python3 scripts/embed_server.py

Requires `pip install fastembed`. Models are downloaded on first start and cached in
~/.cache/fastembed (override with CACHE_DIR).
"""
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from fastembed import TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder

MODEL = os.environ.get("EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
RERANK_MODEL = os.environ.get("RERANK_MODEL", "jinaai/jina-reranker-v2-base-multilingual")
CACHE = os.environ.get("CACHE_DIR", os.path.expanduser("~/.cache/fastembed"))
PORT = int(os.environ.get("PORT", "8099"))

emb = TextEmbedding(model_name=MODEL, cache_dir=CACHE, threads=2)
reranker = TextCrossEncoder(model_name=RERANK_MODEL, cache_dir=CACHE, threads=2)
LOCK = threading.Lock()


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, obj, code=200):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        self._send({"status": "ok", "model": MODEL, "rerank_model": RERANK_MODEL})

    def do_POST(self):
        path = self.path.rstrip("/")
        try:
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n))
            if path in ("/v1/rerank", "/rerank"):
                docs = body.get("documents") or []
                top_n = body.get("top_n") or len(docs)
                with LOCK:
                    scores = [float(s) for s in reranker.rerank(body.get("query") or "", docs)]
                order = sorted(range(len(docs)), key=lambda i: -scores[i])[:top_n]
                self._send({"results": [{"index": i, "relevance_score": scores[i]} for i in order],
                            "model": body.get("model", RERANK_MODEL)})
                return
            if path not in ("/v1/embeddings", "/embeddings"):
                self._send({"error": "not found"}, 404)
                return
            text = body.get("input") or ""
            texts = [text] if isinstance(text, str) else list(text)
            with LOCK:
                vecs = [v.tolist() for v in emb.embed(texts)]
            self._send({"object": "list", "model": MODEL,
                        "data": [{"object": "embedding", "index": i, "embedding": v}
                                 for i, v in enumerate(vecs)]})
        except Exception as e:  # noqa: BLE001
            self._send({"error": str(e)}, 500)


if __name__ == "__main__":
    print(f"embeddings {MODEL} · rerank {RERANK_MODEL} · 127.0.0.1:{PORT}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
