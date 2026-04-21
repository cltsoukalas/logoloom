"""
logoloom — Streamlit Demo App

Framework Alignment and Risk Overview System for Nuclear Reactor Design.

Primary use case: FSAR Coverage Analysis
  → Ingest multiple FSAR sections and produce a regulatory coverage map
    showing which NUREG-0800 SRP criteria are well-addressed, partially
    addressed, or missing (gaps) across the submitted document set.

Secondary use case: Criterion Finder (reviewer mode)
  → Given a specific SRP criterion, surface the FSAR sections that best
    address it, ranked by relevance score.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from logoloom.analyzer import CoverageAnalyzer, CoverageResult
from logoloom.data import load_controls
from logoloom.nexus import Embedder, NexusRanker

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CATALOG_PATH = Path(__file__).parent.parent / "data" / "processed" / "nureg0800_controls.json"
EMBEDDINGS_CACHE = Path(__file__).parent.parent / "models" / "control_embeddings.npy"

GAP_THRESHOLD = 0.25
PARTIAL_THRESHOLD = 0.45

# ---------------------------------------------------------------------------
# Demo FSAR sections — realistic excerpts with intentional coverage gaps
# ---------------------------------------------------------------------------

DEMO_SECTIONS = [
    {
        "id": "FSAR-6.3",
        "title": "Emergency Core Cooling System Performance",
        "text": (
            "The emergency core cooling system (ECCS) is designed to satisfy the acceptance criteria "
            "of 10 CFR 50.46. Following a postulated loss-of-coolant accident (LOCA), the ECCS limits "
            "peak cladding temperature to below 1204°C (2200°F) and maximum cladding oxidation to below "
            "17% equivalent cladding reacted. The high-pressure injection system actuates automatically "
            "on a low pressurizer pressure signal. Low-pressure injection provides long-term core cooling "
            "following system blowdown. Passive accumulators provide rapid injection during the blowdown "
            "phase. Long-term core cooling is maintained by recirculation from the containment sump. "
            "Coolable core geometry is maintained throughout the event sequence."
        ),
    },
    {
        "id": "FSAR-15.6",
        "title": "Loss-of-Coolant Accident Analysis",
        "text": (
            "The large-break and small-break loss-of-coolant accidents are analyzed as design basis "
            "accidents. The analysis demonstrates that peak cladding temperature remains below the 2200°F "
            "limit, cladding oxidation does not exceed 17% ECR, core-wide hydrogen generation remains below "
            "1% from zirconium-water reaction, and the core retains a coolable geometry. Steam generator "
            "tube rupture is analyzed as a separate design basis event; dose consequences at the exclusion "
            "area boundary and low population zone are within 10 CFR 50.67 limits."
        ),
    },
    {
        "id": "FSAR-7.1",
        "title": "Reactor Protection System Design",
        "text": (
            "The reactor protection system (RPS) initiates an automatic reactor trip upon detection of "
            "conditions approaching fuel design limits. Trip parameters include high neutron flux (both "
            "high and low setpoint), high reactor coolant system pressure, low RCS flow, high coolant "
            "outlet temperature, and low-low water level. The RPS is designed with four independent and "
            "redundant channels arranged in a two-out-of-four coincidence logic. Physical and electrical "
            "separation is maintained between channels. The system conforms to IEEE Std 603 for safety "
            "systems. No single active failure prevents a protective action."
        ),
    },
    {
        "id": "FSAR-15.8",
        "title": "Anticipated Transients Without Scram",
        "text": (
            "The plant is equipped with anticipatory trip functions per 10 CFR 50.62. In the event of "
            "an anticipated transient without scram (ATWS), peak reactor coolant system pressure is "
            "limited to below 3200 psia through the action of the turbine trip and feedwater runback "
            "functions. The boron injection system provides long-term reactivity control. Analysis "
            "confirms that fuel damage is limited and containment integrity is maintained. Core power "
            "does not return to pre-transient levels following the transient."
        ),
    },
    {
        "id": "FSAR-19.0",
        "title": "Probabilistic Risk Assessment — Level 1 and Level 2",
        "text": (
            "The Level 1 PRA quantifies accident sequences leading to core damage. Internal events at "
            "full power yield a core damage frequency (CDF) of 2.3 × 10⁻⁷ per reactor year. Dominant "
            "accident sequences include station blackout and small-break LOCA with failure of high-pressure "
            "injection. The Level 2 PRA evaluates containment performance following core damage; the large "
            "early release frequency (LERF) is 1.8 × 10⁻⁸ per reactor year. The PRA has been peer-reviewed "
            "against the ASME/ANS RA-Sa-2009 standard. Risk insights are incorporated into the defense-in-depth "
            "evaluation and technical specification development."
        ),
    },
    {
        "id": "FSAR-4.4",
        "title": "Thermal and Hydraulic Design",
        "text": (
            "The thermal-hydraulic design ensures that departure from nucleate boiling (DNB) does not occur "
            "with at least 95% probability at the 95% confidence level during normal operation and anticipated "
            "operational occurrences. The design limit DNBR is established using NRC-approved correlations. "
            "Fuel centerline temperature remains below the melting point. Coolant flow distribution is "
            "analyzed using a subchannel code. The thermal margin is sufficient to accommodate all "
            "anticipated transients without exceeding the specified acceptable fuel design limits."
        ),
    },
    {
        "id": "FSAR-5.2",
        "title": "Reactor Coolant Pressure Boundary Integrity",
        "text": (
            "The reactor coolant pressure boundary is designed, fabricated, and tested in accordance with "
            "ASME Code Section III, Class 1. Materials are selected for compatibility with the reactor "
            "coolant environment over the design life. In-service inspection is performed per ASME Code "
            "Section XI. Leak-before-break has been demonstrated for the primary coolant piping, excluding "
            "the reactor coolant pump suction and discharge nozzles. Pressure testing requirements of "
            "10 CFR 50 Appendix A GDC 31 and GDC 32 are satisfied."
        ),
    },
    {
        "id": "FSAR-6.2",
        "title": "Containment Functional Design",
        "text": (
            "The containment is designed to withstand the peak pressure and temperature following a "
            "design basis LOCA without loss of structural integrity or leak-tightness. The containment "
            "heat removal system limits long-term containment pressure below the design value. Containment "
            "leakage is within the limits of the technical specifications and 10 CFR 50 Appendix J. "
            "The passive containment cooling system provides long-term heat removal without reliance on "
            "active components or off-site power."
        ),
    },
    # Intentional gaps: SRP-3.9 (Mechanical Systems), SRP-4.2 (Fuel System Design),
    # SRP-4.3 (Nuclear Design), SRP-15.1-15.5 (secondary system transients),
    # SRP-7.3 (ESFAS), SRP-19.1 (PRA adequacy), SRP-5.4 (RCS components), SRP-15.7 (rad release)
]

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="logoloom",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.big-title { font-size: 2rem; font-weight: 700; color: #0d1b2a; margin-bottom: 0; }
.subtitle  { font-size: 0.95rem; color: #666; margin-bottom: 1.5rem; }
.metric-box {
    background: #f8f9fa; border: 1px solid #e0e0e0;
    border-radius: 8px; padding: 1rem; text-align: center;
}
.metric-val { font-size: 2rem; font-weight: 700; }
.metric-lbl { font-size: 0.8rem; color: #888; margin-top: 4px; }
.gap-chip {
    display: inline-block; background: #fee2e2; color: #991b1b;
    border-radius: 12px; padding: 2px 10px; font-size: 0.78rem; font-weight: 600;
}
.partial-chip {
    display: inline-block; background: #fef9c3; color: #854d0e;
    border-radius: 12px; padding: 2px 10px; font-size: 0.78rem; font-weight: 600;
}
.covered-chip {
    display: inline-block; background: #dcfce7; color: #166534;
    border-radius: 12px; padding: 2px 10px; font-size: 0.78rem; font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Model loading (cached)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading embedding model...")
def load_ranker() -> NexusRanker:
    embedder = Embedder()
    controls = load_controls(CATALOG_PATH)
    ranker = NexusRanker(embedder=embedder, controls=controls)
    EMBEDDINGS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    if EMBEDDINGS_CACHE.exists():
        ranker._control_embeddings = np.load(str(EMBEDDINGS_CACHE))
    else:
        ranker.build_control_index(show_progress=False)
        np.save(str(EMBEDDINGS_CACHE), ranker._control_embeddings)
    return ranker

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## ⚛️ logoloom")
    st.caption("Nuclear Framework Alignment Engine")
    st.divider()

    st.markdown("### Framework")
    framework = st.selectbox(
        "Regulatory Framework",
        ["NUREG-0800 (LWR — SRP)", "10 CFR Part 53 (coming soon)"],
        label_visibility="collapsed",
    )

    st.markdown("### Coverage Thresholds")
    gap_t = st.slider("Gap threshold", 0.0, 0.5, GAP_THRESHOLD, 0.05,
                      help="Scores below this are flagged as gaps")
    partial_t = st.slider("Partial threshold", 0.0, 0.7, PARTIAL_THRESHOLD, 0.05,
                          help="Scores below this (but above gap) are 'partial'")

    st.divider()
    st.markdown("### About")
    st.caption(
        "logoloom maps nuclear design documents to NRC regulatory criteria "
        "using semantic embeddings. Primary use: identify regulatory coverage "
        "gaps before NRC submission."
    )
    st.markdown("[GitHub](https://github.com/cltsoukalas/logoloom)")

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.markdown('<div class="big-title">⚛️ logoloom</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Nuclear Regulatory Coverage Analysis — '
    "NUREG-0800 Framework Alignment Engine</div>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_coverage, tab_gaps, tab_criterion, tab_section, tab_arch = st.tabs([
    "📊 Coverage Overview",
    "🚨 Gap Report",
    "🔍 Criterion Finder",
    "📄 Single Section",
    "🏗️ Architecture",
])

# ---------------------------------------------------------------------------
# Shared state: run analysis once, share across tabs
# ---------------------------------------------------------------------------

ranker = load_ranker()

if "sections" not in st.session_state:
    st.session_state.sections = DEMO_SECTIONS
    st.session_state.analysis_done = False

if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False

# ---------------------------------------------------------------------------
# Tab 1: Coverage Overview
# ---------------------------------------------------------------------------

with tab_coverage:
    col_ctrl, col_spacer = st.columns([3, 1])
    with col_ctrl:
        mode = st.radio(
            "Document input",
            ["Use demo FSAR sections", "Add custom sections"],
            horizontal=True,
            label_visibility="collapsed",
        )

    if mode == "Add custom sections":
        with st.expander("➕ Add a document section", expanded=True):
            c1, c2 = st.columns([1, 2])
            with c1:
                new_id = st.text_input("Section ID", placeholder="e.g. FSAR-3.9")
                new_title = st.text_input("Title", placeholder="e.g. Mechanical Components")
            with c2:
                new_text = st.text_area("Section Text", height=120,
                                        placeholder="Paste FSAR section text here...")
            if st.button("Add Section", type="secondary"):
                if new_text.strip():
                    st.session_state.sections.append({
                        "id": new_id or f"FSAR-{len(st.session_state.sections)+1}",
                        "title": new_title or "Untitled Section",
                        "text": new_text,
                    })
                    st.session_state.analysis_done = False
                    st.success(f"Added: {new_title or new_id}")

        st.caption(f"{len(st.session_state.sections)} sections loaded")

        if st.button("Reset to demo data"):
            st.session_state.sections = DEMO_SECTIONS
            st.session_state.analysis_done = False

    run_col, _ = st.columns([1, 3])
    with run_col:
        run_btn = st.button("▶ Run Coverage Analysis", type="primary", use_container_width=True)

    if run_btn or st.session_state.analysis_done:
        with st.spinner("Analyzing regulatory coverage..."):
            analyzer = CoverageAnalyzer(ranker, gap_threshold=gap_t, partial_threshold=partial_t)
            result = analyzer.analyze(st.session_state.sections)
            st.session_state.result = result
            st.session_state.analysis_done = True

        result = st.session_state.result
        summary_df = result.summary_df.copy()
        # Re-apply thresholds if sliders changed since last run
        summary_df["status"] = summary_df["max_score"].apply(
            lambda s: "gap" if s < gap_t else ("partial" if s < partial_t else "covered")
        )

        n_covered = (summary_df["status"] == "covered").sum()
        n_partial = (summary_df["status"] == "partial").sum()
        n_gaps = (summary_df["status"] == "gap").sum()
        n_total = len(summary_df)
        coverage_pct = int((n_covered + 0.5 * n_partial) / n_total * 100)

        # Metrics row
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f"""<div class="metric-box">
                <div class="metric-val" style="color:#166534">{n_covered}</div>
                <div class="metric-lbl">Criteria Covered</div>
            </div>""", unsafe_allow_html=True)
        with m2:
            st.markdown(f"""<div class="metric-box">
                <div class="metric-val" style="color:#854d0e">{n_partial}</div>
                <div class="metric-lbl">Partial Coverage</div>
            </div>""", unsafe_allow_html=True)
        with m3:
            st.markdown(f"""<div class="metric-box">
                <div class="metric-val" style="color:#991b1b">{n_gaps}</div>
                <div class="metric-lbl">Gaps Detected</div>
            </div>""", unsafe_allow_html=True)
        with m4:
            st.markdown(f"""<div class="metric-box">
                <div class="metric-val" style="color:#1d4ed8">{coverage_pct}%</div>
                <div class="metric-lbl">Overall Coverage</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")

        # Coverage bar chart
        color_map = {"gap": "#ef4444", "partial": "#f59e0b", "covered": "#22c55e"}
        plot_df = summary_df.copy()
        plot_df["color"] = plot_df["status"].map(color_map)
        plot_df["label"] = plot_df["id"] + ":  " + plot_df["title"].str[:50]

        fig = go.Figure()
        for status, color in color_map.items():
            mask = plot_df["status"] == status
            fig.add_trace(go.Bar(
                x=plot_df[mask]["max_score"],
                y=plot_df[mask]["label"],
                orientation="h",
                name=status.capitalize(),
                marker_color=color,
                text=plot_df[mask]["max_score"].apply(lambda x: f"{x:.2f}"),
                textposition="outside",
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "Score: %{x:.3f}<br>"
                    "<extra></extra>"
                ),
            ))

        fig.add_vline(x=gap_t, line_dash="dot", line_color="#ef4444",
                      annotation_text="Gap threshold", annotation_position="top right")
        fig.add_vline(x=partial_t, line_dash="dot", line_color="#f59e0b",
                      annotation_text="Partial threshold", annotation_position="top right")

        fig.update_layout(
            title="Regulatory Coverage by SRP Criterion",
            xaxis_title="Max Coverage Score (cosine similarity)",
            yaxis_title="",
            barmode="overlay",
            yaxis={"categoryorder": "total ascending"},
            height=max(400, len(summary_df) * 28),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(l=0, r=80, t=60, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Heatmap
        with st.expander("🗺️ Coverage Heatmap (sections × criteria)", expanded=False):
            heat_df = result.scores_df.T
            heat_df.columns = [s["id"] for s in st.session_state.sections]

            fig_heat = px.imshow(
                heat_df,
                color_continuous_scale="RdYlGn",
                zmin=0, zmax=0.8,
                aspect="auto",
                title="Similarity Scores: SRP Criteria (rows) × FSAR Sections (cols)",
                labels={"color": "Similarity"},
            )
            fig_heat.update_layout(height=550, margin=dict(l=0, r=0, t=50, b=0))
            st.plotly_chart(fig_heat, use_container_width=True)

    else:
        st.info("Click **▶ Run Coverage Analysis** to analyze your FSAR sections against NUREG-0800.")

# ---------------------------------------------------------------------------
# Tab 2: Gap Report
# ---------------------------------------------------------------------------

with tab_gaps:
    if not st.session_state.get("analysis_done"):
        st.info("Run Coverage Analysis first (Coverage Overview tab).")
    else:
        result = st.session_state.result
        summary_df = result.summary_df.copy()
        summary_df["status"] = summary_df["max_score"].apply(
            lambda s: "gap" if s < gap_t else ("partial" if s < partial_t else "covered")
        )
        gaps = summary_df[summary_df["status"] == "gap"].reset_index(drop=True)
        partials = summary_df[summary_df["status"] == "partial"].reset_index(drop=True)

        st.markdown(f"### 🚨 {len(gaps)} Critical Gaps")
        st.caption(
            "These SRP criteria have no FSAR section scoring above the gap threshold. "
            "The NRC will likely issue Requests for Additional Information (RAIs) in these areas."
        )

        if gaps.empty:
            st.success("No critical gaps detected across your document set.")
        else:
            for _, row in gaps.iterrows():
                with st.expander(f"**{row['id']}** — {row['title']}  ⚠️ Score: {row['max_score']:.3f}"):
                    st.markdown(f"**SRP Chapter:** {row['chapter']}")
                    st.markdown(f"**Closest matching section:** `{row['best_section']}` — {row['best_section_title']} (score: {row['max_score']:.3f})")
                    st.markdown("**Acceptance Criteria Summary:**")
                    st.markdown(f"> {row['text'][:600]}{'...' if len(row['text']) > 600 else ''}")
                    st.markdown("**Recommended action:** Add a dedicated FSAR section addressing this criterion.")

        st.markdown(f"### ⚠️ {len(partials)} Partial Coverage Areas")
        st.caption(
            "These criteria have at least one section providing some coverage, "
            "but the semantic match is below the confidence threshold. Review for completeness."
        )

        if partials.empty:
            st.success("No partial coverage areas detected.")
        else:
            for _, row in partials.iterrows():
                with st.expander(f"**{row['id']}** — {row['title']}  ℹ️ Score: {row['max_score']:.3f}"):
                    st.markdown(f"**Best matching section:** `{row['best_section']}` — {row['best_section_title']}")
                    st.markdown(f"**Score:** {row['max_score']:.3f} (partial coverage threshold: {partial_t:.2f})")
                    st.markdown("**Acceptance Criteria Summary:**")
                    st.markdown(f"> {row['text'][:600]}{'...' if len(row['text']) > 600 else ''}")
                    st.markdown("**Recommended action:** Expand the matched section to more explicitly address these acceptance criteria.")

        # Export
        st.divider()
        csv = result.to_export_df().to_csv(index=False)
        st.download_button(
            "⬇️ Export Coverage Report (CSV)",
            data=csv,
            file_name="logoloom_coverage_report.csv",
            mime="text/csv",
        )

# ---------------------------------------------------------------------------
# Tab 3: Criterion Finder (reviewer mode)
# ---------------------------------------------------------------------------

with tab_criterion:
    st.markdown("#### Find FSAR sections for a specific SRP criterion")
    st.caption(
        "Select a regulatory criterion to see which of your FSAR sections best address it — "
        "and how thoroughly. Useful for NRC reviewers or applicants validating specific coverage."
    )

    controls = ranker.controls
    control_options = {f"{c['id']} — {c['title']}": c for c in controls}
    selected_label = st.selectbox("Select SRP Criterion", list(control_options.keys()))
    selected_ctrl = control_options[selected_label]

    st.markdown(f"**Acceptance Criteria:**")
    st.markdown(f"> {selected_ctrl['text']}")
    st.divider()

    if st.session_state.get("analysis_done"):
        result = st.session_state.result
        ctrl_id = selected_ctrl["id"]
        if ctrl_id in result.scores_df.columns:
            df_top = result.top_sections_for_criterion(ctrl_id)
            sections_dict = {s["id"]: s for s in st.session_state.sections}

            st.markdown(f"**FSAR sections ranked by relevance to {ctrl_id}:**")
            for rank, row in enumerate(df_top.itertuples(), 1):
                sec = sections_dict.get(row.section_id, {})
                score = row.score
                with st.expander(
                    f"#{rank}  `{row.section_id}` — {row.section_title}   Score: {score:.3f}",
                    expanded=(rank <= 2)
                ):
                    st.progress(min(score / 0.8, 1.0))
                    st.markdown(f"**Score:** `{score:.4f}`")
                    st.markdown(f"**Section text:**")
                    st.markdown(f"> {sec.get('text', '')[:500]}...")
    else:
        if st.button("▶ Run analysis to enable ranking"):
            pass
        st.info("Run Coverage Analysis first to rank sections by criterion.")

    st.divider()
    st.markdown("##### Or search with custom text (no prior analysis needed)")
    custom_title = st.text_input("Custom title", placeholder="Any topic or system name")
    custom_text = st.text_area("Custom text", height=120,
                               placeholder="Paste any technical text to find the most relevant criteria...")
    if st.button("Search criteria →", type="secondary") and custom_text.strip():
        with st.spinner("Searching..."):
            results = ranker.rank(custom_title or "", custom_text, top_k=8, explain=False)
        for r in results:
            st.markdown(f"**#{r['rank']}  {r['id']}** — {r['title']}  `{r['probability']:.3f}`")
            st.caption(r["text"][:200] + "...")
            st.divider()

# ---------------------------------------------------------------------------
# Tab 4: Single Section Detail
# ---------------------------------------------------------------------------

with tab_section:
    st.markdown("#### Inspect a single FSAR section")
    st.caption("Select any loaded section to see its full alignment profile across all SRP criteria.")

    if st.session_state.get("analysis_done"):
        result = st.session_state.result
        sec_options = {f"{s['id']} — {s['title']}": s for s in st.session_state.sections}
        sel_sec_label = st.selectbox("Select FSAR Section", list(sec_options.keys()))
        sel_sec = sec_options[sel_sec_label]

        st.markdown(f"**Section text:**")
        st.markdown(f"> {sel_sec['text']}")
        st.divider()

        profile_df = result.section_profile(sel_sec["id"])
        plot_df = profile_df.copy()
        plot_df["label"] = plot_df["criterion_id"] + ": " + plot_df["criterion_title"].str[:40]

        fig = px.bar(
            plot_df,
            x="score",
            y="label",
            orientation="h",
            color="score",
            color_continuous_scale="Blues",
            title=f"SRP Criterion Alignment Profile: {sel_sec['id']}",
            labels={"score": "Similarity Score", "label": ""},
        )

        fig.update_layout(
            yaxis={"categoryorder": "total ascending"},
            coloraxis_showscale=False,
            height=max(400, len(plot_df) * 26),
            margin=dict(l=0, r=60, t=50, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Run Coverage Analysis first (Coverage Overview tab).")

# ---------------------------------------------------------------------------
# Tab 5: Architecture
# ---------------------------------------------------------------------------

with tab_arch:
    st.markdown("""
    ## NEXUS-nuclear Architecture

    logoloom adapts the **NEXUS** (Framework Alignment and Risk Overview System) pipeline
    to the nuclear regulatory domain. NEXUS was originally built for cybersecurity compliance
    (NIST 800-53); this adaptation replaces the controls catalog with NUREG-0800 SRP criteria.

    ### Core Pipeline

    ```
    FSAR Section (title + text)
           │
           ▼
    ┌─────────────────────────────┐
    │  Sentence Transformer       │  HuggingFace all-MiniLM-L6-v2
    │  Embedder                   │  L2-normalized, dim=384
    └─────────────┬───────────────┘
                  │  entity_embedding (384,)
                  ▼
    ┌─────────────────────────────┐
    │  Cosine Similarity          │  entity_emb @ control_emb_matrix.T
    │  against all N controls     │  → similarity vector (N_controls,)
    └─────────────┬───────────────┘
                  │
                  ▼  [v0.1: cosine sim used directly]
    ┌─────────────────────────────┐  [v0.2: logistic regression]
    │  Logistic Regression        │  sklearn, trained on weak-supervised pairs
    │  Relevance Ranker           │  → P(relevant) per control
    └─────────────┬───────────────┘
                  │
                  ▼  [v0.2]
    ┌─────────────────────────────┐
    │  SHAP LinearExplainer       │  Per-decision feature attribution
    └─────────────┬───────────────┘
                  │
                  ▼
    Coverage Matrix (N_sections × N_controls)
    Gap Detection | Criterion Ranking | Section Profiling
    ```

    ### Coverage Analysis (primary use case)

    Run across all sections → aggregate scores per criterion → identify gaps.

    | Score | Status | Interpretation |
    |-------|--------|----------------|
    | ≥ 0.45 | ✅ Covered | Section likely addresses this criterion |
    | 0.25–0.45 | ⚠️ Partial | Some overlap, may need expansion |
    | < 0.25 | 🚨 Gap | No section adequately addresses this criterion — RAI risk |

    ### Roadmap

    | Version | Feature |
    |---------|---------|
    | v0.1 | ✅ Coverage analysis, cosine similarity ranking, Streamlit UI |
    | v0.2 | Logistic regression ranker + SHAP explanations (labeled data from ADAMS RAIs) |
    | v0.3 | PULSE-nuclear: compliance classifier (Sufficient / Needs RAI) |
    | v0.4 | 10 CFR Part 53 controls catalog for advanced reactors |
    | v0.5 | REST API + full FSAR PDF ingestion pipeline |

    ### Labeling Strategy (v0.2)

    Training pairs for the logistic regression are derived from:
    1. **NUREG-0800 cross-references** — SRP sections that cite other sections = positive pairs
    2. **NRC ADAMS RAI records** — RAIs identify exactly which SRP criteria an FSAR section
       failed to satisfy, providing negative signal with official ground truth
    """)
