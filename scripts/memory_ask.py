#!/usr/bin/env python3
"""The gate: answer only when a candidate clears a relevance bar.

    python3 scripts/memory_ask.py "how does she take her coffee"
    python3 scripts/memory_ask.py "what is her blood type"        # refuses

Two design rules, both measured (see docs/method.md):

1. The bar is a cross-encoder score, not a retrieval score. Retrieval scores of real and
   unanswerable questions overlap; cross-encoder scores do not.
2. Score the whole top-k and take the best, instead of judging rank 1 alone. Choosing
   among three is easier than judging one, and it raises accuracy as well as safety.

This file is a reference implementation. Wire `search()` to your own store; everything
else stays the same.

`GATE` defaults to −1.9, the value calibrated for this project's corpus. It is **not**
portable: on the bundled demo corpus the same threshold cost 15 points of accuracy.
Run `scripts/benchmark.py <corpus> --calibrate` for yours and pass it via `--gate`/`GATE`.
"""
import argparse
import json
import os
import sys
import urllib.request

import numpy as np

GATE = float(os.environ.get("GATE", "-1.9"))
EMBED_URL = os.environ.get("EMBED_URL", "http://127.0.0.1:8099")
RERANK_URL = os.environ.get("RERANK_URL", "http://127.0.0.1:8099/v1/rerank")


def _post(url, payload):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def _rerank(query, docs):
    hits = _post(RERANK_URL, {"query": query, "documents": docs})["results"]
    return {h["index"]: float(h["relevance_score"]) for h in hits}


def answer_or_refuse(query, candidates, gate=GATE):
    """candidates: [(id, text), ...] already retrieved, best first.

    Returns {answer, id, gate} or {answer: None, reason, gate}.
    """
    if not any("a" <= ch.lower() <= "z" for ch in query):
        return {"answer": None, "gate": None,
                "reason": "no Latin letters: facts are stored in English, translate the query"}
    if not candidates:
        return {"answer": None, "gate": None, "reason": "no candidates"}
    docs = [t for _, t in candidates]
    scores = _rerank(query, docs)
    best = max(range(len(docs)), key=lambda i: scores.get(i, -99.0))
    g = scores.get(best, -99.0)
    if g < gate:
        return {"answer": None, "gate": round(g, 2), "reason": "below gate",
                "would_have_answered": docs[best]}
    return {"answer": docs[best], "id": candidates[best][0], "gate": round(g, 2)}


# --- demo wiring: retrieval over the synthetic corpus, embeddings-only ---------------

def demo_search(query, corpus_path, k=3):
    c = json.load(open(corpus_path))
    texts = [f["text"] for f in c["facts"]]
    vecs = np.array([d["embedding"] for d in _post(EMBED_URL + "/v1/embeddings",
                                                   {"input": texts})["data"]], dtype="float32")
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    q = np.array(_post(EMBED_URL + "/v1/embeddings", {"input": [query]})["data"][0]["embedding"],
                 dtype="float32")
    q /= np.linalg.norm(q)
    order = np.argsort(-(vecs @ q))[:k]
    return [(c["facts"][i]["id"], texts[i]) for i in order]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--corpus", default="data/demo_corpus.json")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    if not os.path.exists(a.corpus):
        sys.exit(f"corpus not found: {a.corpus} (run from the repo root)")
    out = answer_or_refuse(a.query, demo_search(a.query, a.corpus, a.k))
    if a.json:
        print(json.dumps(out, indent=1))
    elif out["answer"]:
        print(f"[{out['gate']:+.2f}] {out['answer']}")
    else:
        print(f"[{out['gate']}] no confident answer ({out['reason']})")


if __name__ == "__main__":
    main()
