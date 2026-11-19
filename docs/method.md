# Method

## Why a benchmark at all

Without a number, "memory upgrade" is a feeling. Every change in this project had to
answer one question: **does the right fact come back first, and does the system stay
quiet when it has nothing?** Two metrics, one adversarial set.

## The set

- **26 answerable questions** over a 126-fact store, split roughly 50/50 between
  keyword-shaped ("what model is configured") and paraphrase-shaped ("how should my
  replies be styled").
- **5 unanswerable questions** — plausible questions whose answers are *not* in the
  store (blood type, bank, a child's name, a cluster he doesn't run, a subscription
  price). A good system returns nothing for all five.

Expected ids are verified to exist before a run; a probe pointing at a deleted fact is
dropped rather than silently counted as a miss.

## Three systems, one query at a time

1. **keyword only** — FTS5, the honest baseline most memory tools are built on.
2. **the stack** — keywords + embeddings + HRR, the production configuration.
3. **the stack + gate** — same retrieval, but a cross-encoder decides whether to answer.

Reporting all three is the point: it shows what the embeddings actually buy, and what
the gate costs in latency.

## The gate

A gate is a second model that reads `(question, candidate fact)` together and returns a
relevance score. It is not a threshold on the retrieval score — those overlap:

```
retrieval scores   answerable 0.29–0.42   unanswerable 0.26–0.32   (no usable cut-off)
cross-encoder      answerable −1.53..+1.70   unanswerable −2.29..−3.18   (clean gap)
```

Calibrate on your own corpus in four steps:

1. Score ~14 answerable and ~5 unanswerable questions through the reranker.
2. Print both ranges. If they overlap, the corpus or the reranker is wrong — do not pick
   a threshold anyway.
3. Put the gate in the middle of the gap.
4. Re-run the benchmark. Expect 0 false answers and *no loss* of correct answers.

### The gate is corpus-specific — prove it before shipping

The same code and reranker, run on a second corpus, needed a different threshold and
punished the copied one:

```
production corpus (126 facts)   gate −1.9    92% recall@1   0/5 false answers
demo corpus, production gate    gate −1.9    77% recall@1   0/5   ← refuses correct answers
demo corpus, re-calibrated      gate −3.11   96% recall@1   0/5
```

Two lessons, both in `scripts/benchmark.py`:

- `--calibrate` prints the ranges and the implied gate for *your* data.
- The benchmark warns, and exits non-zero-adjacent, when the gate costs correct answers.
- A wide gap is the signal that a threshold is safe. The demo corpus's gap is 0.05 —
  eight times narrower than the production corpus's 0.76 — which is a warning that 26
  facts are not enough to calibrate on.

## Choosing among the top three

The naive gate judges only rank 1. Because the reranker returns a score per document, it
can also choose which of the top three is best — a free accuracy gain:

```
rank-1 only, ungated    85% recall@1
rank-1 gate             85% recall@1   0/5 false answers
best-of-3 with the gate 92% recall@1   0/5 false answers
```

The same component that was added for safety improved accuracy, because "which of these
three is the answer" is an easier question than "is this the answer".

## Rules that keep the numbers honest

- **Never tune on the test set.** Probes were added once (14 → 26) because one question
  was deciding the verdict; after that the set was frozen.
- **Publish the failures.** A rejected experiment with a number is more valuable than a
  feature with an adjective.
- **State the sample size.** Each question here is worth ~4 points; a 7-point gap is one
  question and should not be called a win.
- **Latency is a metric.** A 20-second recall is not a memory system, it is a batch job.
