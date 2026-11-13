# LAGCD
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


## Project files

`lagcd/{data,encoding,lcot,har,mcr,uac,model,losses,metrics,engine,llm,utils}.py`, `scripts/{make_demo,train,evaluate,predict}.py`, `tests/test_smoke.py`, `config.yaml`, `requirements.txt`.
