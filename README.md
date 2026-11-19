# A memory system that says "I don't know"

Agent memory that recalls by meaning and refuses when the match is weak. Every change
gets a number attached, including the changes that made it worse.

```
26 questions + 5 unanswerable, on a 126-fact store

keyword only (FTS5)      85% recall@1    5/5 false answers
hybrid stack             85% recall@1    4/5 false answers    112 ms
hybrid + relevance gate  92% recall@1    0/5 false answers    112 ms + gate
Hindsight 1.2.1 (local)  64% recall@1    0/5 false answers    ~20,000 ms
```

## The threshold does not travel

The gate that works on the production store is -1.9. Run unchanged against the bundled
26-fact demo corpus, it scored 77% recall@1, because it refused correct answers. After
calibrating on that corpus:

```
python3 scripts/benchmark.py data/demo_corpus.json --calibrate
  answerable   min -3.09  max +1.38
  unanswerable min -3.51  max -3.14
  suggested gate: -3.11

python3 scripts/benchmark.py data/demo_corpus.json --gate -3.11
  retrieval      92% recall@1   5/5 false answers
  + gate -3.11   96% recall@1   0/5 false answers
```

Same code, same reranker, different data, different threshold. Calibrate it, do not copy
it. The demo corpus leaves a gap of only 0.05, which is a sign that 26 facts are too few
to calibrate against. The production store left 0.76.

## Why a gate instead of a score cutoff

A memory that always returns something is worse than one that admits it has nothing. In a
personal assistant, a confident wrong answer about your manager or your bills is a failure
you never notice. The agent repeats it and you believe it.

Similarity scores cannot tell you when to stay quiet. Answerable and unanswerable questions
overlap (0.29 to 0.42 against 0.26 to 0.32), so any cutoff that removes the junk removes
real answers too. A cross-encoder that reads the question and the candidate together does
separate them:

```
answerable    -1.53 .. +1.70
unanswerable  -2.29 .. -3.18     gate at -1.9
```

The same reranker also picks among the top three instead of judging only the first, and
that is what moved recall@1 from 85% to 92%. A safety component turned into an accuracy
gain because choosing one of three is easier than judging one candidate alone.

## What was tried and rejected

| change | result | verdict |
|---|---|---|
| Hindsight 1.2.1 (local, CPU) | 64% recall@1, 86% @3, ~20 s/query, 124 facts became 288 stored items | slower and less accurate |
| reranker reorders the whole pool | recall@1 79% to 71%, 4x latency | rejected |
| English-only reranker as the gate | scored a correct Arabic match at -4.21 | replaced with a multilingual one |
| bigger embeddings alone | top-1 down, top-3 up, 2x latency | kept only with the gate |

These are in the history because "we tried it and it was worse" is the most useful thing a
benchmark can publish.

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

Everything runs locally: a 1 GB embedding model and a 1.1 GB reranker on the same CPU-only
box as the agent. There is no memory API and no per-query cost.

## Repo layout

```
docs/method.md          how the benchmark is built and why it has 26 questions
scripts/memory_ask.py   reference implementation of the gate
scripts/benchmark.py    corpus in, recall@1/@3 and false answers out
scripts/embed_server.py local embeddings and /v1/rerank service (stdlib HTTP)
data/demo_corpus.json   synthetic corpus, 26 probes and 5 unanswerable questions
```

## Reproduce

```bash
python3 scripts/embed_server.py &                 # 127.0.0.1:8099
python3 scripts/benchmark.py data/demo_corpus.json
python3 scripts/memory_ask.py "what is the demo subject's blood type"   # refuses
```

The demo corpus carries its own calibrated gate in a `gate` field, and both scripts read
it. That is deliberate: a threshold copied from another store refuses correct answers, so
the corpus travels with the number that was measured on it. Override with `--gate`, or
recompute with `--calibrate`.

The published numbers come from the production plugin, which also uses keyword search and
trust scoring. `benchmark.py` approximates that with its embedding leg, and the gate
behaves the same way in both.

## Limits

- 26 questions is a small set. One question is worth about 4 points, so read the gaps as
  direction rather than precision.
- Cross-language retrieval is unreliable. Facts and queries are stored in English, and a
  query without Latin letters is refused with an explanation instead of returning nothing.
- 44 entity names in the author's store are unconfirmed guesses, and there is no queue to
  merge them.

MIT licensed.
