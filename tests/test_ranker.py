"""Tests for logoloom.nexus.ranker — NexusRanker"""

import numpy as np
import pytest

from logoloom.nexus.embedder import Embedder
from logoloom.nexus.ranker import NexusRanker

SEED_CONTROLS = [
    {"id": "SRP-15.6", "title": "Decrease in Reactor Coolant Inventory",
     "text": "Loss-of-coolant accident analysis. Peak cladding temperature limits, cladding oxidation limits.",
     "chapter": "15", "section": "15.6", "framework": "NUREG-0800"},
    {"id": "SRP-6.3", "title": "Emergency Core Cooling System",
     "text": "ECCS design: high-pressure injection, low-pressure injection, accumulators. 10 CFR 50.46 acceptance criteria.",
     "chapter": "6", "section": "6.3", "framework": "NUREG-0800"},
    {"id": "SRP-19.0", "title": "Probabilistic Risk Assessment",
     "text": "Level 1 PRA: core damage frequency. Level 2: large early release frequency. ASME/ANS RA-Sa-2009.",
     "chapter": "19", "section": "19.0", "framework": "NUREG-0800"},
    {"id": "SRP-7.1", "title": "Instrumentation and Controls",
     "text": "Reactor protection system, ESFAS, single-failure criterion, IEEE Std 603.",
     "chapter": "7", "section": "7.1", "framework": "NUREG-0800"},
    {"id": "SRP-4.4", "title": "Thermal and Hydraulic Design",
     "text": "DNBR limits, fuel centerline temperature, coolant flow, thermal margin.",
     "chapter": "4", "section": "4.4", "framework": "NUREG-0800"},
]


@pytest.fixture(scope="module")
def embedder():
    return Embedder()


@pytest.fixture(scope="module")
def ranker(embedder):
    r = NexusRanker(embedder=embedder, controls=SEED_CONTROLS)
    r.build_control_index()
    return r


# ── Construction ──────────────────────────────────────────────────────────────

def test_empty_controls_raises():
    with pytest.raises(ValueError, match="controls list must not be empty"):
        NexusRanker(embedder=Embedder(), controls=[])


def test_build_control_index_sets_embeddings(ranker):
    assert ranker._control_embeddings is not None
    assert ranker._control_embeddings.shape == (len(SEED_CONTROLS), ranker.embedder.embedding_dim)


# ── Similarity scores ─────────────────────────────────────────────────────────

def test_get_similarity_scores_returns_correct_shape(ranker):
    sims = ranker.get_similarity_scores("ECCS", "Emergency core cooling system performance")
    assert sims.shape == (len(SEED_CONTROLS),)


def test_get_similarity_scores_range(ranker):
    sims = ranker.get_similarity_scores("Reactor protection", "Trip signals, redundant channels")
    assert sims.min() >= -1.0
    assert sims.max() <= 1.0


def test_get_similarity_scores_no_index_raises():
    embedder = Embedder()
    r = NexusRanker(embedder=embedder, controls=SEED_CONTROLS)
    with pytest.raises(RuntimeError, match="Control index not built"):
        r.get_similarity_scores("title", "text")


# ── Rank output structure ─────────────────────────────────────────────────────

def test_rank_returns_list(ranker):
    results = ranker.rank("ECCS", "Emergency core cooling following LOCA", top_k=3)
    assert isinstance(results, list)


def test_rank_respects_top_k(ranker):
    for k in (1, 3, 5):
        results = ranker.rank("ECCS", "Emergency core cooling", top_k=k)
        assert len(results) == k


def test_rank_top_k_exceeds_controls_clipped(ranker):
    results = ranker.rank("ECCS", "text", top_k=9999)
    assert len(results) == len(SEED_CONTROLS)


def test_rank_result_has_required_keys(ranker):
    required = {"rank", "id", "title", "text", "chapter", "section",
                "framework", "cosine_sim", "probability"}
    results = ranker.rank("ECCS", "Emergency core cooling system", top_k=3)
    for r in results:
        assert required.issubset(set(r.keys())), f"Missing keys: {required - set(r.keys())}"


def test_rank_ranks_are_sequential(ranker):
    results = ranker.rank("PRA", "Core damage frequency probabilistic risk assessment", top_k=5)
    ranks = [r["rank"] for r in results]
    assert ranks == list(range(1, len(results) + 1))


def test_rank_probabilities_descending(ranker):
    results = ranker.rank("ECCS", "Emergency core cooling following LOCA", top_k=5)
    probs = [r["probability"] for r in results]
    assert probs == sorted(probs, reverse=True), "Probabilities should be in descending order"


def test_rank_eccs_text_returns_eccs_control_in_top2(ranker):
    """
    Semantic sanity check: ECCS text should rank SRP-6.3 (ECCS) or SRP-15.6 (LOCA)
    in the top 2 positions.
    """
    results = ranker.rank(
        "Emergency Core Cooling System",
        "The ECCS limits peak cladding temperature to below 1204C following a "
        "design basis loss-of-coolant accident. High-pressure injection actuates "
        "on low pressurizer pressure. Acceptance criteria per 10 CFR 50.46.",
        top_k=3,
    )
    top2_ids = {r["id"] for r in results[:2]}
    assert top2_ids & {"SRP-6.3", "SRP-15.6"}, (
        f"Expected ECCS/LOCA controls in top 2, got: {top2_ids}"
    )


def test_rank_pra_text_returns_pra_control_at_top(ranker):
    results = ranker.rank(
        "Probabilistic Risk Assessment",
        "Core damage frequency is 2.3e-7 per reactor year. Large early release "
        "frequency is 1.8e-8. Peer reviewed against ASME/ANS RA-Sa-2009.",
        top_k=2,
    )
    top_id = results[0]["id"]
    assert top_id == "SRP-19.0", f"Expected SRP-19.0 at rank 1, got {top_id}"


# ── Logistic regression path ───────────────────────────────────────────────────

def test_fit_and_rank_with_trained_model(ranker, embedder):
    """Fit a minimal LR model and verify rank() uses it without error."""
    n_samples = 20
    rng = np.random.default_rng(42)
    X = rng.random((n_samples, len(SEED_CONTROLS))).astype(np.float32)
    y = rng.integers(0, 2, size=n_samples)

    # Ensure both classes present
    y[0] = 0
    y[1] = 1

    ranker.fit(X, y)
    results = ranker.rank("ECCS", "Emergency core cooling system", top_k=3, explain=True)
    assert len(results) == 3
    assert all("overall_prob" in r for r in results)
    assert all("shap_value" in r for r in results)


def test_fit_wrong_feature_count_raises(ranker):
    X = np.random.rand(10, 99)   # wrong number of features
    y = np.random.randint(0, 2, 10)
    with pytest.raises(ValueError, match="features but catalog has"):
        ranker.fit(X, y)


def test_fit_mismatched_labels_raises(ranker):
    X = np.random.rand(10, len(SEED_CONTROLS))
    y = np.random.randint(0, 2, 8)   # wrong number of labels
    with pytest.raises(ValueError, match="samples but y has"):
        ranker.fit(X, y)


def test_fit_wrong_X_dims_raises(ranker):
    X = np.random.rand(10)   # 1D instead of 2D
    y = np.random.randint(0, 2, 10)
    with pytest.raises(ValueError, match="2D"):
        ranker.fit(X, y)


# ── Save / Load round-trip ────────────────────────────────────────────────────

def test_save_load_round_trip(ranker, tmp_path):
    ranker.save(tmp_path / "test_model")
    loaded = NexusRanker.load(tmp_path / "test_model", ranker.embedder)

    assert len(loaded.controls) == len(ranker.controls)
    assert loaded._control_embeddings is not None
    np.testing.assert_allclose(
        loaded._control_embeddings, ranker._control_embeddings, atol=1e-5
    )


def test_load_missing_controls_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        NexusRanker.load(tmp_path / "nonexistent", Embedder())
