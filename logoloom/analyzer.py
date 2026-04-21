"""
CoverageAnalyzer — service layer for regulatory gap analysis.

Sits between NexusRanker and any consumer (Streamlit, FastAPI, CLI, tests).
Both the UI and the API call this; neither duplicates the analysis logic.

Usage
-----
    from logoloom.analyzer import CoverageAnalyzer
    from logoloom.nexus import Embedder, NexusRanker
    from logoloom.data import load_controls

    ranker = NexusRanker(Embedder(), load_controls())
    ranker.build_control_index()

    analyzer = CoverageAnalyzer(ranker)
    result = analyzer.analyze(sections)

    print(result.summary_df)
    print(result.gap_df)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .nexus.ranker import NexusRanker

logger = logging.getLogger(__name__)

DEFAULT_GAP_THRESHOLD     = 0.25
DEFAULT_PARTIAL_THRESHOLD = 0.45


@dataclass
class CoverageResult:
    """
    Full output of a coverage analysis run.

    Attributes
    ----------
    scores_df   : DataFrame (N_sections × N_controls) of raw cosine similarity scores.
    summary_df  : DataFrame (N_controls) with per-criterion max score, best section, status.
    gap_df      : Subset of summary_df where status == "gap".
    partial_df  : Subset of summary_df where status == "partial".
    covered_df  : Subset of summary_df where status == "covered".
    sections    : Original list of section dicts passed to analyze().
    gap_threshold     : Threshold used to classify gaps.
    partial_threshold : Threshold used to classify partials.
    """

    scores_df: pd.DataFrame
    summary_df: pd.DataFrame
    sections: list[dict]
    gap_threshold: float
    partial_threshold: float

    # Derived views — populated post-init
    gap_df: pd.DataFrame = field(init=False)
    partial_df: pd.DataFrame = field(init=False)
    covered_df: pd.DataFrame = field(init=False)

    def __post_init__(self) -> None:
        self.gap_df     = self.summary_df[self.summary_df["status"] == "gap"].reset_index(drop=True)
        self.partial_df = self.summary_df[self.summary_df["status"] == "partial"].reset_index(drop=True)
        self.covered_df = self.summary_df[self.summary_df["status"] == "covered"].reset_index(drop=True)

    @property
    def n_controls(self) -> int:
        return len(self.summary_df)

    @property
    def n_gaps(self) -> int:
        return len(self.gap_df)

    @property
    def n_partial(self) -> int:
        return len(self.partial_df)

    @property
    def n_covered(self) -> int:
        return len(self.covered_df)

    @property
    def coverage_pct(self) -> float:
        """Weighted coverage percentage (covered=1.0, partial=0.5, gap=0.0)."""
        if self.n_controls == 0:
            return 0.0
        return (self.n_covered + 0.5 * self.n_partial) / self.n_controls * 100

    def top_sections_for_criterion(self, criterion_id: str) -> pd.DataFrame:
        """
        Return all sections ranked by their score for a given criterion ID.
        """
        if criterion_id not in self.scores_df.columns:
            raise KeyError(f"Criterion '{criterion_id}' not found in scores matrix.")
        col = self.scores_df[criterion_id].sort_values(ascending=False)
        section_lookup = {s["id"]: s for s in self.sections}
        rows = []
        for sec_id, score in col.items():
            sec = section_lookup.get(sec_id, {})
            rows.append({
                "section_id": sec_id,
                "section_title": sec.get("title", ""),
                "score": float(score),
            })
        return pd.DataFrame(rows)

    def section_profile(self, section_id: str) -> pd.DataFrame:
        """
        Return all criteria ranked by their score for a given section.
        """
        if section_id not in self.scores_df.index:
            raise KeyError(f"Section '{section_id}' not found in scores matrix.")
        row = self.scores_df.loc[section_id].sort_values(ascending=False)
        ctrl_lookup = {}
        if not self.summary_df.empty:
            ctrl_lookup = self.summary_df.set_index("id")[["title", "chapter"]].to_dict("index")
        rows = []
        for ctrl_id, score in row.items():
            info = ctrl_lookup.get(ctrl_id, {})
            rows.append({
                "criterion_id": ctrl_id,
                "criterion_title": info.get("title", ""),
                "chapter": info.get("chapter", ""),
                "score": float(score),
            })
        return pd.DataFrame(rows)

    def to_export_df(self) -> pd.DataFrame:
        """Flat DataFrame suitable for CSV export."""
        df = self.summary_df[["id", "title", "chapter", "max_score", "status",
                               "best_section", "best_section_title"]].copy()
        df.columns = [
            "Criterion ID", "Title", "Chapter", "Score",
            "Status", "Best Section ID", "Best Section Title",
        ]
        return df


class CoverageAnalyzer:
    """
    Service class: runs the NEXUS pipeline across a set of FSAR sections and
    produces a CoverageResult.

    Parameters
    ----------
    ranker            : NexusRanker with a built control index.
    gap_threshold     : Scores below this → "gap".
    partial_threshold : Scores below this (but above gap) → "partial".
    """

    def __init__(
        self,
        ranker: NexusRanker,
        gap_threshold: float = DEFAULT_GAP_THRESHOLD,
        partial_threshold: float = DEFAULT_PARTIAL_THRESHOLD,
    ) -> None:
        if ranker._control_embeddings is None:
            raise RuntimeError(
                "Ranker has no control index. Call ranker.build_control_index() first."
            )
        self.ranker = ranker
        self.gap_threshold = gap_threshold
        self.partial_threshold = partial_threshold

    def _classify(self, score: float) -> str:
        if score < self.gap_threshold:
            return "gap"
        if score < self.partial_threshold:
            return "partial"
        return "covered"

    def analyze(self, sections: list[dict]) -> CoverageResult:
        """
        Run coverage analysis across all sections.

        Parameters
        ----------
        sections : list of dicts with keys "id", "title", "text".

        Returns
        -------
        CoverageResult with scores matrix and per-criterion summary.
        """
        if not sections:
            raise ValueError("sections list must not be empty")

        controls = self.ranker.controls
        n_controls = len(controls)
        control_ids = [c["id"] for c in controls]
        section_ids = [s["id"] for s in sections]

        # Validate sections have required keys
        for i, sec in enumerate(sections):
            for key in ("id", "title", "text"):
                if key not in sec:
                    raise ValueError(f"Section at index {i} is missing required key '{key}'")

        # Compute scores matrix (N_sections × N_controls)
        scores = np.zeros((len(sections), n_controls), dtype=np.float32)
        for i, sec in enumerate(sections):
            scores[i, :] = self.ranker.get_similarity_scores(sec["title"], sec["text"])

        scores_df = pd.DataFrame(scores, index=section_ids, columns=control_ids)

        # Per-control summary
        summary_rows = []
        for j, ctrl in enumerate(controls):
            col_scores = scores[:, j]
            best_idx = int(np.argmax(col_scores))
            max_score = float(col_scores[best_idx])
            summary_rows.append({
                "id": ctrl["id"],
                "title": ctrl.get("title", ""),
                "chapter": ctrl.get("chapter", ""),
                "framework": ctrl.get("framework", "NUREG-0800"),
                "text": ctrl.get("text", ""),
                "max_score": max_score,
                "best_section": sections[best_idx]["id"],
                "best_section_title": sections[best_idx]["title"],
                "status": self._classify(max_score),
            })

        summary_df = (
            pd.DataFrame(summary_rows)
            .sort_values("max_score", ascending=True)
            .reset_index(drop=True)
        )

        logger.info(
            "Coverage analysis: %d sections × %d controls — "
            "covered=%d, partial=%d, gap=%d",
            len(sections), n_controls,
            (summary_df["status"] == "covered").sum(),
            (summary_df["status"] == "partial").sum(),
            (summary_df["status"] == "gap").sum(),
        )

        return CoverageResult(
            scores_df=scores_df,
            summary_df=summary_df,
            sections=sections,
            gap_threshold=self.gap_threshold,
            partial_threshold=self.partial_threshold,
        )
