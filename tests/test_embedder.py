"""Tests for logoloom.nexus.embedder"""

import numpy as np
import pytest

from logoloom.nexus.embedder import Embedder


@pytest.fixture(scope="module")
def embedder():
    return Embedder()


def test_encode_one_returns_1d_array(embedder):
    vec = embedder.encode_one("ECCS Design", "The emergency core cooling system...")
    assert isinstance(vec, np.ndarray)
    assert vec.ndim == 1
    assert vec.shape[0] == embedder.embedding_dim


def test_encode_one_is_l2_normalized(embedder):
    vec = embedder.encode_one("Some title", "Some text about nuclear design")
    norm = float(np.linalg.norm(vec))
    assert abs(norm - 1.0) < 1e-5, f"Expected unit norm, got {norm}"


def test_encode_one_empty_title(embedder):
    """Empty title should not raise — falls back to text only."""
    vec = embedder.encode_one("", "Some text about reactor coolant pressure")
    assert vec.shape[0] == embedder.embedding_dim


def test_encode_batch_returns_correct_shape(embedder):
    pairs = [
        ("ECCS", "Emergency core cooling system design"),
        ("PRA", "Probabilistic risk assessment level 1"),
        ("Containment", "Containment pressure and temperature response"),
    ]
    matrix = embedder.encode_batch(pairs)
    assert matrix.shape == (3, embedder.embedding_dim)


def test_encode_batch_rows_are_l2_normalized(embedder):
    pairs = [("Title A", "Text A"), ("Title B", "Text B")]
    matrix = embedder.encode_batch(pairs)
    norms = np.linalg.norm(matrix, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)


def test_encode_batch_single_pair_matches_encode_one(embedder):
    title, text = "Nuclear Design", "Reactivity control and shutdown margin"
    vec_one = embedder.encode_one(title, text)
    vec_batch = embedder.encode_batch([(title, text)])
    np.testing.assert_allclose(vec_one, vec_batch[0], atol=1e-5)


def test_similar_texts_higher_similarity_than_dissimilar(embedder):
    """Nuclear-domain semantics: ECCS text should be closer to ECCS criteria than to PRA."""
    eccs_text = embedder.encode_one(
        "Emergency Core Cooling",
        "Peak cladding temperature must remain below 1204C following LOCA"
    )
    eccs_criteria = embedder.encode_one(
        "SRP-6.3",
        "ECCS acceptance criteria per 10 CFR 50.46: PCT below 2200F, cladding oxidation below 17%"
    )
    pra_criteria = embedder.encode_one(
        "SRP-19.0",
        "Core damage frequency and large early release frequency from probabilistic risk assessment"
    )
    sim_relevant = float(eccs_text @ eccs_criteria)
    sim_irrelevant = float(eccs_text @ pra_criteria)
    assert sim_relevant > sim_irrelevant, (
        f"Expected ECCS text to be closer to ECCS criteria ({sim_relevant:.3f}) "
        f"than to PRA criteria ({sim_irrelevant:.3f})"
    )
