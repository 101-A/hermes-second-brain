#!/usr/bin/env python3
"""Self-contained memory benchmark: corpus in, recall@1/@3 + false answers out.

    python3 scripts/benchmark.py data/demo_corpus.json
    python3 scripts/benchmark.py data/demo_corpus.json --no-gate   # skip the reranker
    EMBED_URL=http://127.0.0.1:8099 RERANK_URL=http://127.0.0.1:8099/v1/rerank ...

Needs the local service from scripts/embed_server.py (or any OpenAI-shaped embeddings
endpoint and Cohere-shaped rerank endpoint).

Corpus format (see data/demo_corpus.json):
    {"facts": [{"id": 1, "text": "..."}],
     "probes": [{"q": "...", "expect": 1}],
     "abstain": ["..."]}
"""
import argparse
import json
import os
import sys
import urllib.request

import numpy as np

EMBED_URL = os.environ.get("EMBED_URL", "http://127.0.0.1:8099")
RERANK_URL = os.environ.get("RERANK_URL", "http://127.0.0.1:8099/v1/rerank")
GATE = float(os.environ.get("GATE", "-1.9"))


def post(url, payload, timeout=120):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def embed(texts):
    out = []
    for i in range(0, len(texts), 32):
        out += [d["embedding"] for d in post(EMBED_URL + "/v1/embeddings",
                                             {"input": texts[i:i + 32]})["data"]]
    v = np.array(out, dtype="float32")
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def rerank(query, docs):
    hits = post(RERANK_URL, {"query": query, "documents": docs})["results"]
    return {h["index"]: float(h["relevance_score"]) for h in hits}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus")
    ap.add_argument("--no-gate", action="store_true")
    ap.add_argument("--calibrate", action="store_true",
                    help="print the score ranges and the gate they imply, then exit")
    ap.add_argument("--gate", type=float, default=GATE)
    ap.add_argument("--k", type=int, default=3)
    a = ap.parse_args()

    c = json.load(open(a.corpus))
    facts = c["facts"]
    texts = [f["text"] for f in facts]
    ids = [f["id"] for f in facts]
    F = embed(texts)

    def top(q, k):
        qv = embed([q])[0]
        order = np.argsort(-(F @ qv))[:k]
        return [ids[i] for i in order], [texts[i] for i in order]

    def best_score(q, docs):
        return max(rerank(q, docs).values(), default=-99.0)

    if a.calibrate:
        ans = [best_score(p["q"], top(p["q"], a.k)[1]) for p in c["probes"]]
        una = [best_score(q, top(q, a.k)[1]) for q in c.get("abstain", [])]
        print(f"answerable   min {min(ans):+.2f}  max {max(ans):+.2f}  (n={len(ans)})")
        if una:
            print(f"unanswerable min {min(una):+.2f}  max {max(una):+.2f}  (n={len(una)})")
            gap = (max(una) + min(ans)) / 2
            if min(ans) <= max(una):
                print(f"OVERLAP — no gate separates this corpus; do not ship a threshold")
            else:
                print(f"suggested gate: {gap:.2f}   (gap between {max(una):+.2f} and {min(ans):+.2f})")
        return

    r1 = r3 = g1 = 0
    for p in c["probes"]:
        got, docs = top(p["q"], a.k)
        r1 += got[0] == p["expect"]
        r3 += p["expect"] in got
        if not a.no_gate:
            scores = rerank(p["q"], docs)
            best = max(range(len(docs)), key=lambda i: scores.get(i, -99.0))
            if scores.get(best, -99.0) >= a.gate:
                g1 += got[best] == p["expect"]

    fp = gfp = 0
    for q in c.get("abstain", []):
        got, docs = top(q, a.k)
        fp += bool(got)
        if not a.no_gate:
            gfp += best_score(q, docs) >= a.gate

    n, m = len(c["probes"]), len(c.get("abstain", []))
    print(f"facts {len(facts)} · probes {n} · unanswerable {m}")
    print(f"  retrieval      recall@1 {r1}/{n} ({r1/n:.0%})  recall@{a.k} {r3}/{n} ({r3/n:.0%})"
          f"  false answers {fp}/{m}")
    if not a.no_gate:
        print(f"  + gate {a.gate}   recall@1 {g1}/{n} ({g1/n:.0%})"
              f"                      false answers {gfp}/{m}")
        if g1 < r1:
            print("  warning: the gate cost you correct answers — recalibrate with --calibrate",
                  file=sys.stderr)


if __name__ == "__main__":
    main()
