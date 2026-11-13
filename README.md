# LAGCD Python reproduction (research reconstruction)

This project **reconstructs the main trainable architecture** described in the provided manuscript, *LLM-Augmented Governance and Compliance Decision: A Framework for Intelligent AI Regulation and Legal Reasoning*. It is **not the authors' code** and does **not** claim to reproduce the manuscript's reported benchmark numbers.

## What is implemented

| Paper component | Location | Notes |
|---|---|---|
| Statutory Parser | `lagcd/encoding.py` | 3-layer 256-d Transformer, 4 heads, 2 graph aggregation layers; hierarchical adjacency provided or approximated |
| Legal Knowledge Encoding | `lagcd/encoding.py`, `lagcd/data.py` | statutory and context streams; simple deterministic hashing tokens; *not pretrained LegalBERT or 2.3M-case corpus* |
| LCoT | `lagcd/lcot.py`, `lagcd/llm.py` | 5-step answers, encoded by a trainable encoder and a 1-layer GRU; optional Claude API |
| HAR | `lagcd/har.py` | 3 layers, 4 heads, 4 distinct cross-attention streams, sigmoid jurisdiction-conditioned routing, residual/FFN |
| MCR | `lagcd/mcr.py` | pairwise compatibility, -0.3 threshold, 3-class resolution, review flag |
| UAC | `lagcd/uac.py` | classification, risk score 0–100, learned positive temperature |
| Training and ablation | `lagcd/engine.py`, `scripts/train.py` | AdamW, cosine schedule, warmup, early stopping; `--ablate har` etc. |
| Evaluation | `lagcd/metrics.py`, `scripts/evaluate.py` | macro F1, macro AUROC, MAE, ECE, confusion matrix, per-class F1 |

## Quick start

```bash
python -m pip install -r requirements.txt
python scripts/make_demo.py --n 120
# For a fast CPU test, temporarily reduce the dimensions / epochs in config.yaml.
python scripts/train.py --config config.yaml --data data/demo --out runs/demo
python scripts/evaluate.py --checkpoint runs/demo/best.pt --test data/demo/test.jsonl
python scripts/predict.py --checkpoint runs/demo/best.pt --input data/demo/example.json
```

For a short end-to-end CPU smoke run, use `python scripts/make_demo.py --n 24`, then `python scripts/train.py --config config.smoke.yaml --data data/demo --out runs/smoke`, followed by `python scripts/evaluate.py --checkpoint runs/smoke/best.pt --test data/demo/test.jsonl`.

Generate a valid prediction input JSON by copying a complete line from `data/demo/test.jsonl` to `data/demo/example.json`. Alternatively create a JSON object with keys `scenario`, `statutes`, `provisions`, `case_law`, `policy`, `entities`, `norms`, `jurisdiction_id`, `lcot_answers`. Example:

```json
{"scenario":"AI tool handling sensitive medical records", "statutes":"Article 1: Document legal safeguards.", "provisions":["Article 1: Document legal safeguards."], "jurisdiction_id":1, "case_law":"", "policy":"Privacy policy", "entities":["controller"], "norms":[{"text":"Protect personal data", "jurisdiction_id":1}], "lcot_answers":["Applicable rule","Documentation","Check safeguards","Review exceptions","Needs review"]}
```

### Optional Anthropic generation

```bash
pip install anthropic
export ANTHROPIC_API_KEY='your-key'
# edit config.yaml: api.provider: anthropic
python scripts/predict.py --checkpoint runs/demo/best.pt --input data/demo/example.json --generate
```

Model availability and actual API model IDs must be checked with the provider. API requests have cost, network and security implications. Avoid sending private or identifying case records to external APIs without authorization. Mock mode emits generic **unverified** placeholders and cannot serve as legal reasoning evidence.

### Dataset format

Bring your own **train.jsonl**, **val.jsonl** and **test.jsonl** to a folder and point `--data` to it. Each row is a JSON object:

- `id`, `scenario`, `statutes`: strings (scenario is used by API mode, statutes supply input provisions)
- `provisions`: list of article/paragraph texts; `edges`: optional `[parent_index,child_index]` pairs
- `case_law`, `policy`, `entities`: strings or lists of strings
- `norms`: list of `{ "text": ..., "jurisdiction_id": 0 }`, with integers in `[0, num_jurisdictions)`
- `jurisdiction_id`: numeric ID; keep mapping consistent between train and test
- `lcot_answers`: exactly five strings; use a trusted fixed API for preprocessing if desired
- `label`: integer 0–3 (`compliant`, `minor_risk`, `major_risk`, `non_compliant`)
- `risk`: numeric in 0–100; optional, ignored for MSE if missing
- `rationale`: optional trusted reference rationale used for **embedding-alignment surrogate**, not LLM token loss
- `conflict_label`: optional int 0,1,2 for first/second jurisdiction prevails, irresolvable; omission disables conflict supervision

The dataset loader does not download or create legal compliance ground truth. **`scripts/make_demo.py` creates synthetic software-check data and must never be used to report paper performance.** Ensure no train/test leakage. For 5-seed trials use `seed` 42,123,256,512,1024 and independent output directories.

## Gaps and scientific limits

1. The paper does not supply its complete original training examples, 2.3M legal-document index, expert-labeled jurisdiction conflict pairs, or trainable backbone feature interface. Consequently, the LegalBERT embedder, actual RAG FAISS corpus, production legal-entity RGCN and original annotations are **not reproduced**. This implementation gives modular proxy representations, not exact preprocessing.
2. The manuscript says custom modules comprise ~420M trainable parameters, but its specified layer dimensions alone do not establish that count. The implemented parameter count is printed by `scripts/train.py`; we do not artificially pad the model to 420M.
3. LLM API weights are fixed. An API-produced response cannot provide gradient for token-level `L_gen`. To keep training executable, the code explicitly uses a differentiable *rationale embedding-alignment surrogate* (only if reference rationales exist). This is not mathematically the manuscript's stated next-token cross-entropy.
4. The manuscript defines ECE but exact hard-binned ECE is nondifferentiable. Training uses a smooth-bin surrogate; evaluation uses conventional hard-bin ECE. Temperature is jointly optimized rather than separately post-fit on validation data, another documented approximation.
5. The paper's external dataset links should be independently checked for format, annotation provenance, and whether they actually contain the described 4-class scenarios and numeric risk labels. Do not claim reported F1 91.2%, AUROC 93.5%, 64 ms, or 420M parameters without independently reproducing them.
6. Pairwise conflict prediction and simple norm pooling reproduce the *shape* of MCR, but full expert legal precedence rules and normative constraint satisfaction require additional datasets and symbolic supervision.
7. All predictions are research outputs; they are **not legal advice**. Human legal review is necessary, especially for unresolved cross-jurisdiction conflicts.

## Project files

`lagcd/{data,encoding,lcot,har,mcr,uac,model,losses,metrics,engine,llm,utils}.py`, `scripts/{make_demo,train,evaluate,predict}.py`, `tests/test_smoke.py`, `config.yaml`, `requirements.txt`.
