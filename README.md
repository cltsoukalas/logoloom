# ⚛️ logoloom

**Framework Alignment and Risk Overview System for Nuclear Reactor Design**

logoloom maps nuclear design document sections to the most relevant NRC regulatory criteria using semantic embeddings and a logistic regression ranker — automating the initial framework alignment step in reactor design review.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## What it does

Given any section of a nuclear design document (FSAR section, safety analysis, design basis argument), logoloom returns the top-K most relevant NRC Standard Review Plan (NUREG-0800) criteria with:

- **Relevance probability** — how likely this document section addresses the given criterion
- **Cosine similarity** — raw semantic proximity between the document text and criterion text
- **SHAP explanation** — which features drove the alignment decision, for human-auditable review

## Architecture

logoloom is a nuclear domain adaptation of the **NEXUS** (Framework Alignment and Risk Overview System) architecture.

```
Entity of Interest (FSAR section title + text)
        │
        ▼
Sentence Transformer (all-MiniLM-L6-v2)
        │  L2-normalized embeddings
        ▼
Cosine Similarity against all N framework controls
        │  similarity vector (N,)
        ▼
Logistic Regression Ranker (sklearn)
        │  P(relevant) per control
        ▼
SHAP LinearExplainer
        │  feature importance
        ▼
Top-K ranked SRP criteria
```

The controls catalog (NUREG-0800 SRP criteria) is a swappable JSON file. Switching to 10 CFR Part 53 performance objectives for advanced reactors requires only a new catalog — no architecture changes.

## Regulatory Coverage

| Framework | Status | Reactor Types |
|-----------|--------|---------------|
| NUREG-0800 (Standard Review Plan, LWR Edition) | ✅ Active | PWR, BWR |
| 10 CFR Part 53 (Advanced Reactor Framework) | 🔜 Planned v0.2 | SMR, non-LWR, Microreactor |

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Build the NUREG-0800 controls catalog

```bash
python scripts/build_nureg0800_catalog.py
```

### 3. Run the Streamlit app

```bash
streamlit run app/streamlit_app.py
```

### 4. Use as a library

```python
from logoloom.nexus import Embedder, NexusRanker
from logoloom.data import load_controls

embedder = Embedder()
controls = load_controls()

ranker = NexusRanker(embedder=embedder, controls=controls)
ranker.build_control_index()

results = ranker.rank(
    title="Emergency Core Cooling System",
    text=(
        "The ECCS is designed to limit peak cladding temperature to below 1204°C "
        "and cladding oxidation to below 17% ECR following a design basis LOCA. "
        "High-pressure injection actuates on low pressurizer pressure signal..."
    ),
    top_k=10,
)

for r in results:
    print(f"#{r['rank']:2d}  {r['id']}  {r['probability']:.3f}  {r['title']}")
```

## Project Structure

```
logoloom/
├── logoloom/
│   ├── nexus/
│   │   ├── embedder.py       # SentenceTransformer wrapper
│   │   └── ranker.py         # Core NEXUS pipeline (similarity + LR + SHAP)
│   ├── analyzer.py           # CoverageAnalyzer service layer
│   └── data/
│       └── loader.py         # Controls catalog loader
├── app/
│   └── streamlit_app.py      # Interactive demo
├── scripts/
│   └── build_nureg0800_catalog.py   # Data pipeline
├── data/
│   ├── raw/                  # Downloaded NRC documents
│   └── processed/
│       └── nureg0800_controls.json  # Structured controls catalog
├── notebooks/                # Methodology walkthrough
└── tests/
```

## Labeling Strategy

The logistic regression ranker is trained on weakly supervised pairs derived from:

1. **NUREG-0800 internal cross-references** — SRP sections that cite other sections provide natural positive training pairs
2. **NRC Requests for Additional Information (RAIs)** — RAIs against submitted FSARs identify where a document section did *not* adequately address a criterion (negative signal), available via [NRC ADAMS](https://www.nrc.gov/reading-rm/adams.html)

## Roadmap

- [x] v0.1 — NEXUS-nuclear with NUREG-0800, cosine similarity ranking, Streamlit demo
- [ ] v0.2 — Logistic regression ranker trained on ADAMS-derived labels + SHAP explanations
- [ ] v0.3 — PULSE-nuclear: compliance classifier (Review Required / Sufficient) for full FSAR sections
- [ ] v0.4 — 10 CFR Part 53 controls catalog for advanced reactor support
- [ ] v0.5 — REST API + batch processing for full FSAR ingestion

## Background

Nuclear reactor design review is a document-intensive process. The NRC's Standard Review Plan (NUREG-0800) defines acceptance criteria across 21 chapters that a Final Safety Analysis Report must address. Manually mapping FSAR sections to applicable SRP criteria is time-consuming and difficult to audit consistently.

logoloom automates the initial alignment step — surfacing which regulatory criteria a given document section addresses, which are underserved, and which deserve closer expert review. It is not a replacement for expert NRC review; it is a tool to focus and accelerate it.

## License

MIT
