"""Tests for logoloom.analyzer — CoverageAnalyzer and CoverageResult"""

import numpy as np
import pytest

from logoloom.analyzer import CoverageAnalyzer, CoverageResult
from logoloom.nexus.embedder import Embedder
from logoloom.nexus.ranker import NexusRanker

CONTROLS = [
    {"id": "SRP-15.6", "title": "Decrease in Reactor Coolant Inventory",
     "text": "LOCA analysis. Peak cladding temperature limits.",
     "chapter": "15", "framework": "NUREG-0800"},
    {"id": "SRP-6.3", "title": "Emergency Core Cooling System",
     "text": "ECCS: HPI, LPI, accumulators. 10 CFR 50.46.",
     "chapter": "6", "framework": "NUREG-0800"},
    {"id": "SRP-19.0", "title": "Probabilistic Risk Assessment",
     "text": "CDF, LERF. ASME/ANS RA-Sa-2009.",
     "chapter": "19", "framework": "NUREG-0800"},
]

SECTIONS = [
    {"id": "FSAR-6.3", "title": "Emergency Core Cooling System",
     "text": "The ECCS limits peak cladding temperature below 1204C. High-pressure injection "
             "actuates on low pressurizer pressure. 10 CFR 50.46 acceptance criteria are met."},
    {"id": "FSAR-19", "title": "Probabilistic Risk Assessment",
     "text": "Core damage frequency is 2.3e-7/yr. Large early release frequency is 1.8e-8/yr. "
             "PRA peer-reviewed to ASME/ANS RA-Sa-2009."},
    # Intentionally no section strongly covering SRP-15.6 → should be a partial/gap
]


@pytest.fixture(scope="module")
def analyzer():
    embedder = Embedder()
    ranker = NexusRanker(embedder=embedder, controls=CONTROLS)
    ranker.build_control_index()
    return CoverageAnalyzer(ranker, gap_threshold=0.25, partial_threshold=0.45)


@pytest.fixture(scope="module")
def result(analyzer):
    return analyzer.analyze(SECTIONS)


# ── CoverageAnalyzer construction ──────────────────────────────────────────────

def test_analyzer_raises_without_index():
    embedder = Embedder()
    ranker = NexusRanker(embedder=embedder, controls=CONTROLS)
    # No build_control_index()
    with pytest.raises(RuntimeError, match="no control index"):
        CoverageAnalyzer(ranker)


def test_analyzer_raises_on_empty_sections(analyzer):
    with pytest.raises(ValueError, match="sections list must not be empty"):
        analyzer.analyze([])


def test_analyzer_raises_on_section_missing_key(analyzer):
    bad_sections = [{"id": "X", "title": "T"}]   # missing "text"
    with pytest.raises(ValueError, match="missing required key 'text'"):
        analyzer.analyze(bad_sections)


# ── CoverageResult shape ───────────────────────────────────────────────────────

def test_result_scores_df_shape(result):
    assert result.scores_df.shape == (len(SECTIONS), len(CONTROLS))


def test_result_summary_df_has_all_controls(result):
    assert len(result.summary_df) == len(CONTROLS)


def test_result_status_values_are_valid(result):
    valid = {"gap", "partial", "covered"}
    assert set(result.summary_df["status"]).issubset(valid)


def test_result_counts_sum_to_total(result):
    assert result.n_gaps + result.n_partial + result.n_covered == result.n_controls


def test_result_scores_in_valid_range(result):
    assert result.scores_df.values.min() >= -1.0
    assert result.scores_df.values.max() <= 1.0


# ── CoverageResult derived views ───────────────────────────────────────────────

def test_coverage_pct_between_0_and_100(result):
    assert 0.0 <= result.coverage_pct <= 100.0


def test_top_sections_for_criterion_returns_sorted(result):
    df = result.top_sections_for_criterion("SRP-19.0")
    scores = df["score"].tolist()
    assert scores == sorted(scores, reverse=True)


def test_top_sections_for_criterion_unknown_id_raises(result):
    with pytest.raises(KeyError, match="SRP-UNKNOWN"):
        result.top_sections_for_criterion("SRP-UNKNOWN")


def test_section_profile_returns_sorted(result):
    df = result.section_profile("FSAR-6.3")
    scores = df["score"].tolist()
    assert scores == sorted(scores, reverse=True)


def test_section_profile_unknown_id_raises(result):
    with pytest.raises(KeyError, match="FSAR-UNKNOWN"):
        result.section_profile("FSAR-UNKNOWN")


def test_export_df_has_expected_columns(result):
    df = result.to_export_df()
    expected = {"Criterion ID", "Title", "Chapter", "Score",
                "Status", "Best Section ID", "Best Section Title"}
    assert expected.issubset(set(df.columns))


# ── Semantic sanity ────────────────────────────────────────────────────────────

def test_eccs_section_best_covers_eccs_or_loca_criterion(result):
    """FSAR-6.3 (ECCS) should best cover SRP-6.3 (ECCS) or SRP-15.6 (LOCA)."""
    df = result.section_profile("FSAR-6.3")
    top_criterion = df.iloc[0]["criterion_id"]
    assert top_criterion in {"SRP-6.3", "SRP-15.6"}, (
        f"Expected ECCS/LOCA criterion at top for ECCS section, got {top_criterion}"
    )


def test_pra_section_best_covers_pra_criterion(result):
    """FSAR-19 (PRA) should best cover SRP-19.0 (PRA)."""
    df = result.section_profile("FSAR-19")
    top_criterion = df.iloc[0]["criterion_id"]
    assert top_criterion == "SRP-19.0", (
        f"Expected SRP-19.0 at top for PRA section, got {top_criterion}"
    )
