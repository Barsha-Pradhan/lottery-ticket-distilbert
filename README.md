# Lottery Ticket Hypothesis Pruning: DistilBERT on GLUE

This repository benchmarks **DistilBERT** using the **Iterative Magnitude Pruning (IMP) / Lottery Ticket Hypothesis** pipeline across all nine **GLUE** tasks. Each task-specific script (`sst-2`, `qnli.py`, `qqp.py`, `wnli.py`, `mnli.py`, `mrpc.py`, `sts-b.py`, `rte.py`, `cola.py`) runs the pruning pipeline and reports a common set of efficiency and quality metrics.

Related work: [Model-compression-intel-prune-OFA](https://github.com/nalinipradash/Model-compression-intel-prune-OFA)

## Repository Structure

| File | GLUE Task |
|---|---|
| `sst-2` | SST-2 (Sentiment) |
| `qnli.py` | QNLI (Question NLI) |
| `qqp.py` | QQP (Quora Question Pairs) |
| `wnli.py` | WNLI (Winograd NLI) |
| `mnli.py` | MNLI (Multi-Genre NLI) |
| `mrpc.py` | MRPC (Paraphrase) |
| `sts-b.py` | STS-B (Semantic Similarity) |
| `rte.py` | RTE (Textual Entailment) |
| `cola.py` | CoLA (Linguistic Acceptability) |

## Metrics Tracked

For each task, the following are reported:

- **Memory Size** (MB)
- **Latency** (ms)
- **Accuracy** (%)
- **Bits** (quantization precision)
- **Precision / Recall / F1-score** (%)
- **Throughput** (samples/sec)
- **Energy Consumption** (J)

## Results — DistilBERT

| Dataset | Memory Size (MB) | Latency (ms) | Accuracy (%) | Bits | Precision | Recall | F1-score | Throughput | Energy (J) |
|---|---|---|---|---|---|---|---|---|---|
| SST-2 | 255.41 | 1.5543 | 90.83 | 32 | 90.87 | 90.83 | 90.82 | – | – |
| QNLI | 255.41 | 2.7351 | 89.07 | 32 | 89.09 | 89.07 | 89.07 | 365.62 | 52.8129 |
| QQP | 255.41 | 1.8466 | 83.56 | 32 | 83.99 | 83.56 | 83.69 | 541.53 | 8.3283 |
| WNLI | 255.41 | 2.4981 | 33.80 | 32 | 24.58 | 33.80 | 28.47 | 400.30 | 0.3076 |
| MNLI | 255.41 | 2.3188 | 68.14 | 32 | 68.13 | 68.14 | 68.13 | 431.27 | 3.4988 |
| MRPC | 255.41 | 2.5715 | 85.29 | 32 | 85.25 | 85.29 | 84.70 | 388.87 | 1.8001 |
| STS-B | 255.41 | 1.6712 | N/A | 32 | N/A | N/A | N/A | 598.38 | 2.3659 |
| RTE | 255.41 | 3.7992 | 54.87 | 32 | 54.60 | 54.87 | 52.62 | 263.21 | 1.8361 |
| CoLA | 255.41 | 0.9956 | 80.63 | 32 | 80.08 | 80.63 | 76.69 | 1004.43 | 2.1248 |

> STS-B is a regression task; standard classification accuracy/precision/recall/F1 are not applicable (typically evaluated with Pearson/Spearman correlation instead).

## Key Observations

- **Model size** is constant across tasks (255.41 MB), since it depends on the pruned architecture, not the task.
- **Strongest tasks**: SST-2, QNLI, MRPC, and QQP, where DistilBERT stays in the 83–91% accuracy range.
- **Weakest tasks**: WNLI and RTE, both small, adversarially-constructed GLUE tasks where accuracy drops to the mid-30s/mid-50s — a known difficulty for smaller models on these datasets.
- **Latency/Throughput**: fastest on CoLA (0.9956 ms, ~1004 samples/sec) and slowest on RTE (3.7992 ms); this generally tracks dataset/sequence characteristics rather than a fixed cost.
- **Energy consumption**: highest on QNLI (52.81 J) and lowest on WNLI (0.31 J), with no simple relationship to accuracy or latency alone.

## Setup

```bash
git clone https://github.com/Barsha-Pradhan/lottery-ticket-distilbert.git
cd lottery-ticket-distilbert
pip install torch transformers datasets scikit-learn
```

## Usage

Run any task script directly, e.g.:

```bash
python cola.py
python mrpc.py
```

Each script loads the corresponding GLUE task, applies iterative magnitude pruning (lottery ticket pipeline) to DistilBERT, fine-tunes/evaluates it, and reports the metrics table above.

## Notes

- All results are at **32-bit (FP32)** precision; no quantization is applied in this benchmark.
- SST-2 throughput and energy consumption are not yet recorded (marked "–") — pending re-run.
- For a complementary compression benchmark (pruning, OFA-based methods), see [Model-compression-intel-prune-OFA](https://github.com/nalinipradash/Model-compression-intel-prune-OFA).

