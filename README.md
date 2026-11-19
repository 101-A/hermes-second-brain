# A Second Brain That Says "I Don't Know"

An agent memory system that recalls by meaning, refuses to answer when the match is
weak, and is measured against its own baseline instead of vibes.

Most "AI memory" projects ship a retrieval stack and call it done. This one keeps a
number attached to every change — and publishes the changes that made it *worse*.

```
26 questions + 5 unanswerable, measured on a 126-fact store

keyword only (FTS5)      85% recall@1    5/5 false answers
hybrid stack             85% recall@1    4/5 false answers    112 ms
hybrid + relevance gate  92% recall@1    0/5 false answers    112 ms + gate
Hindsight 1.2.1 (local)  64% recall@1    0/5 false answers    ~20,000 ms
```

### Thresholds do not transfer (measured)

The production gate is **−1.9**. Applied unchanged to the bundled 26-fact demo corpus it
scored **77%** recall@1 — it refused correct answers. Re-calibrated on that corpus:

```
python3 scripts/benchmark.py data/demo_corpus.json --calibrate
  answerable   min -3.09  max +1.38
  unanswerable min -3.51  max -3.14
  suggested gate: -3.11

python3 scripts/benchmark.py data/demo_corpus.json --gate -3.11
  retrieval      92% recall@1   5/5 false answers
  + gate -3.11   96% recall@1   0/5 false answers
```

Same code, same reranker, different corpus, different threshold. **Calibrate; never copy
a gate.** Note the demo's gap is only 0.05 wide — a 26-fact corpus is too small to trust
for calibration. The production corpus had 0.76.

## The problem with "it remembers everything"

A memory that always returns something is worse than one that admits ignorance. In a
personal assistant, a confidently wrong fact about your manager, your bills, or your
kids is a failure you cannot see — the agent repeats it, and you believe it.

Two things fixed that here.

**1. A gate, not a threshold on the retrieval score.** Keyword and embedding scores
overlap between real and unanswerable questions (measured: answerable 0.29–0.42 vs junk
0.26–0.32), so no cut-off on them can abstain. A cross-encoder separates cleanly:

```
answerable top-1    −1.53 .. +1.70
unanswerable top-1  −2.29 .. −3.18      → gate at −1.9
```

**2. Let the reranker pick among the top three, not just judge the top one.** This is
where an accuracy *gain* came from the same component that was supposed to only guard
against errors: 85% → 92% recall@1, with false answers going to zero.

## What was tried and rejected

| change | result | verdict |
|---|---|---|
| Hindsight 1.2.1 (local, CPU) | 64% recall@1, 86% @3, ~20 s/query, 124 facts → 288 stored items | slower and less accurate |
| reranker reorders the whole pool | recall@1 79% → 71%, 4x latency | rejected |
| English-only reranker as the gate | scored a *correct* Arabic match −4.21 | replaced with a multilingual one |
| bigger embeddings alone | top-1 −, top-3 +, 2x latency | kept only with the gate |

Every one of these is in this repo's history because "we tried it and it was worse" is
the most useful thing a benchmark can say.

## Architecture

```
Hermes Agent
   └── holographic memory plugin
         ├── SQLite fact store (facts, versions, trust, entities, keywords)
         ├── FTS5 keyword leg
         ├── local embeddings  (paraphrase-multilingual-mpnet, 768-dim, 127.0.0.1)
         ├── HRR / vector legs
         └── memory_ask.py  →  cross-encoder gate (jina-reranker-v2 multilingual)
                                 ├─ above gate → answer
                                 └─ below gate → "I don't know"
```

Everything runs locally: a 1 GB embedding model and a 1.1 GB reranker on the same
2-vCPU box as the agent. No memory API, no per-query cost.

## Repo layout

```
docs/method.md          how the benchmark is built and why it is 26 questions
scripts/memory_ask.py   reference implementation of the gate
scripts/benchmark.py    self-contained: corpus in, recall@1/@3 + false answers out
scripts/embed_server.py local embeddings + /v1/rerank service (stdlib HTTP)
data/demo_corpus.json   synthetic corpus, 26 probes + 5 unanswerable
```

## Reproduce

```bash
python3 scripts/embed_server.py &                 # 127.0.0.1:8099
python3 scripts/benchmark.py data/demo_corpus.json
python3 scripts/memory_ask.py "what is the demo subject's blood type"   # refuses
```

The published numbers come from the production plugin (FTS + embeddings + HRR + trust),
which `benchmark.py` approximates with its embedding leg; the gate behaves identically.

## Limits, stated

- 26 questions is small: one question ≈ 4 points. Treat gaps as directional.
- Cross-lingual retrieval is unreliable. Facts and queries are stored in English; a
  non-Latin query is refused with an explanation instead of returning nothing.
- 44 entity names in the author's store are provisional guesses; there is no merge queue.

MIT licensed.
