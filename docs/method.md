# Method

## Why measure at all

Without a number, "memory upgrade" is a feeling. Every change in this project had to answer
one question: does the right fact come back first, and does the system stay quiet when it
has nothing? Two metrics, one adversarial set.

## The set

26 answerable questions over a 126-fact store, split roughly half keyword-shaped ("what
model is configured") and half paraphrase-shaped ("how should my replies be styled").

5 unanswerable questions, meaning plausible questions whose answers are not in the store:
blood type, bank, a child's name, a cluster the subject does not run, a subscription price.
A good system returns nothing for all five.

Expected ids are verified before a run. A probe pointing at a deleted fact is dropped
rather than silently counted as a miss.

## Three configurations, one query at a time

1. keyword only. FTS5, the baseline most memory tools are built on.
2. the stack. Keywords plus embeddings plus HRR, the production setup.
3. the stack plus gate. Same retrieval, and a cross-encoder decides whether to answer.

Reporting all three shows what the embeddings actually buy and what the gate costs in
latency.

## The gate

A gate is a second model that reads the question and a candidate fact together and returns
a relevance score. It is not a cutoff on the retrieval score, because those overlap:

```
retrieval scores   answerable 0.29 to 0.42   unanswerable 0.26 to 0.32   (no usable cutoff)
cross-encoder      answerable -1.53 to 1.70  unanswerable -2.29 to -3.18 (clean gap)
```

Calibrating takes four steps:

1. Score about 14 answerable and 5 unanswerable questions through the reranker.
2. Print both ranges. If they overlap, the corpus or the reranker is wrong, and picking a
   threshold anyway just hides it.
3. Put the gate in the middle of the gap.
4. Run the benchmark again. Expect zero false answers and no loss of correct ones.

### Prove the gate on your own corpus

The same code and reranker, run on a second corpus, needed a different threshold and
punished the copied one:

```
production corpus (126 facts)   gate -1.9    92% recall@1   0/5 false answers
demo corpus, production gate    gate -1.9    77% recall@1   0/5   (refuses correct answers)
demo corpus, re-calibrated      gate -3.11   96% recall@1   0/5
```

Both tools for this live in `scripts/benchmark.py`. `--calibrate` prints the ranges and the
gate they imply for your data, and the script warns when a gate costs correct answers.

A wide gap is the signal that a threshold is safe to use. The demo corpus leaves 0.05,
eight times narrower than the production corpus's 0.76, which is the tool telling you that
26 facts are not enough to calibrate against.

## Choosing among the top three

The naive gate judges only rank 1. Because the reranker returns a score for every document,
it can also choose which of the top three is best, and that is free accuracy:

```
rank-1 only, ungated    85% recall@1
rank-1 gate             85% recall@1   0/5 false answers
best-of-3 with the gate 92% recall@1   0/5 false answers
```

The component that was added for safety improved accuracy, because "which of these three is
the answer" is an easier question than "is this the answer".

## Rules that keep the numbers honest

- Never tune on the test set. The probes were added once, going from 14 to 26, because one
  question was deciding the verdict. After that the set was frozen.
- Publish the failures. A rejected experiment with a number is worth more than a feature
  with an adjective.
- State the sample size. Each question here is worth about 4 points, so a 7-point gap is
  one question and should not be reported as a win.
- Treat latency as a metric. A 20-second recall is not a memory system, it is a batch job.
