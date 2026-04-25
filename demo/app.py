"""Gradio demo for AdaptiveInterviewEnv — HuggingFace Spaces entry point."""
import os
import sys
import json
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HUB_REPO = os.getenv("SCORER_HUB_REPO", "")

# ---------------------------------------------------------------------------
# Global state (loaded once at startup)
# ---------------------------------------------------------------------------

_scorer = None
_env = None
_load_error = None


def _init():
    global _scorer, _env, _load_error
    try:
        from adaptive_interview_env.scorer import Scorer
        from adaptive_interview_env.env import AdaptiveInterviewEnv
        from adaptive_interview_env.reward import RewardFunction
        from adaptive_interview_env.models import RewardWeights, CalibrationRef
        from adaptive_interview_env.question_generator import QuestionBank, generate_question

        # Load scorer
        model_path = HUB_REPO or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "outputs", "scorer_final",
        )
        _scorer = Scorer(model_name_or_path=model_path)

        # Build env (no student — human plays student role)
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        qb_path = os.path.join(repo_root, "adaptive_interview_env", "data", "question_bank.json")
        fallback_bank = QuestionBank.from_json(qb_path) if os.path.exists(qb_path) else None

        def question_gen_fn(**kwargs):
            return generate_question(fallback_bank=fallback_bank, **kwargs)

        # Load calibration refs so calibration_score can be non-zero in the demo
        cal_path = os.path.join(repo_root, "adaptive_interview_env", "data", "calibration_refs.json")
        calibration_refs = []
        if os.path.exists(cal_path):
            try:
                with open(cal_path) as f:
                    raw = json.load(f)
                for item in raw:
                    calibration_refs.append(CalibrationRef(
                        question=item.get("question", ""),
                        answer=item.get("answer", ""),
                        ground_truth_scores=item.get("ground_truth_scores", {}),
                    ))
                logger.info(f"Loaded {len(calibration_refs)} calibration refs for reward.")
            except Exception as e:
                logger.warning(f"Failed to load calibration refs: {e}")

        weights = RewardWeights(calibration=0.5, improvement=0.3, consistency=0.2)
        reward_fn = RewardFunction(weights=weights, calibration_refs=calibration_refs)

        _env = AdaptiveInterviewEnv(
            question_generator=question_gen_fn,
            reward_function=reward_fn,
            max_steps=20,
        )
        logger.info("Demo initialised successfully.")
    except Exception as e:
        _load_error = str(e)
        logger.error(f"Demo init failed: {e}")


_init()

# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Skill radar chart
# ---------------------------------------------------------------------------

# Human-readable labels for skill dimensions (shown on radar axes)
DIM_LABELS = {
    "correctness": "Correctness",
    "edge_case_coverage": "Edge Cases",
    "complexity_analysis": "Complexity",
    "tradeoff_reasoning": "Tradeoffs",
    "communication_clarity": "Clarity",
}

# One-line tooltips for each dimension
DIM_TOOLTIPS = {
    "correctness": "Is the answer factually right?",
    "edge_case_coverage": "Did they handle tricky inputs and boundary cases?",
    "complexity_analysis": "Did they reason about time/space complexity?",
    "tradeoff_reasoning": "Did they weigh design tradeoffs?",
    "communication_clarity": "Is the explanation clear and well-structured?",
}


def _score_color(score: float) -> str:
    """Green >= 0.7, amber 0.4-0.7, red < 0.4."""
    if score >= 0.7:
        return "#16a34a"  # green-600
    if score >= 0.4:
        return "#d97706"  # amber-600
    return "#dc2626"      # red-600


def _make_radar_chart(
    skill_profile_dict: dict,
    baseline_dict: dict = None,
    target_dimension: str = None,
    pristine: bool = False,
):
    """Render an easy-to-read skill radar.

    - 0-100 scale for at-a-glance reading
    - Baseline (dotted) shows where the session started; solid shows current
    - Per-vertex score labels color-coded by level (green/amber/red)
    - Target dimension (what the Scorer is probing next) gets a highlight ring
    - pristine=True renders an unscored placeholder before step 1 so we don't
      mislead the viewer with the 0.5 prior as if it were a measurement
    """
    try:
        import plotly.graph_objects as go

        # Stable dimension order
        dims = list(DIM_LABELS.keys())
        labels = [DIM_LABELS[d] for d in dims]

        if pristine:
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=[50] * (len(dims) + 1),
                theta=labels + [labels[0]],
                mode="lines",
                line=dict(color="#cbd5e1", width=1, dash="dot"),
                name="Awaiting first answer",
                hoverinfo="skip",
                showlegend=True,
            ))
            fig.add_trace(go.Scatterpolar(
                r=[0] * len(dims),
                theta=labels,
                mode="markers+text",
                marker=dict(size=10, color="#cbd5e1", line=dict(color="white", width=2)),
                text=["—"] * len(dims),
                textposition="top center",
                textfont=dict(size=13, color="#94a3b8"),
                showlegend=False,
                hoverinfo="skip",
            ))
            fig.add_annotation(
                text="<span style='font-size:13px;color:#94a3b8'>"
                     "Submit an answer<br>to begin scoring</span>",
                x=0.5, y=0.5, xref="paper", yref="paper",
                showarrow=False, align="center",
            )
            fig.update_layout(
                polar=dict(
                    bgcolor="#f8fafc",
                    radialaxis=dict(
                        visible=True, range=[0, 100],
                        tickvals=[20, 40, 60, 80, 100],
                        ticktext=["20", "40", "60", "80", "100"],
                        tickfont=dict(size=10, color="#94a3b8"),
                        gridcolor="#e2e8f0", linecolor="#e2e8f0",
                    ),
                    angularaxis=dict(
                        tickfont=dict(size=13, color="#0f172a"),
                        gridcolor="#e2e8f0", linecolor="#cbd5e1",
                    ),
                ),
                showlegend=True,
                legend=dict(
                    orientation="h", yanchor="bottom", y=-0.15,
                    xanchor="center", x=0.5, font=dict(size=11),
                ),
                margin=dict(l=60, r=60, t=30, b=60),
                height=420,
                paper_bgcolor="white",
            )
            return fig

        current = [skill_profile_dict.get(d, 0.5) * 100 for d in dims]
        baseline = (
            [baseline_dict.get(d, 0.5) * 100 for d in dims]
            if baseline_dict else None
        )

        # Close the polygons
        labels_closed = labels + [labels[0]]
        current_closed = current + [current[0]]

        fig = go.Figure()

        # Baseline trace (start of session)
        if baseline is not None:
            baseline_closed = baseline + [baseline[0]]
            fig.add_trace(go.Scatterpolar(
                r=baseline_closed,
                theta=labels_closed,
                mode="lines",
                line=dict(color="#94a3b8", width=1.5, dash="dot"),
                name="Start of session",
                hovertemplate="<b>%{theta}</b><br>start: %{r:.0f}<extra></extra>",
            ))

        # Current trace (filled area)
        fig.add_trace(go.Scatterpolar(
            r=current_closed,
            theta=labels_closed,
            mode="lines",
            line=dict(color="#2563eb", width=2.5),
            fill="toself",
            fillcolor="rgba(37, 99, 235, 0.18)",
            name="Current skill",
            hovertemplate="<b>%{theta}</b><br>current: %{r:.0f}<extra></extra>",
        ))

        # Per-vertex colored score markers + text labels
        marker_colors = [_score_color(v / 100) for v in current]
        fig.add_trace(go.Scatterpolar(
            r=current,
            theta=labels,
            mode="markers+text",
            marker=dict(size=12, color=marker_colors, line=dict(color="white", width=2)),
            text=[f"{v:.0f}" for v in current],
            textposition="top center",
            textfont=dict(size=12, color="#0f172a"),
            showlegend=False,
            hoverinfo="skip",
        ))

        # Highlight the dimension the Scorer is currently targeting
        if target_dimension and target_dimension in DIM_LABELS:
            idx = dims.index(target_dimension)
            fig.add_trace(go.Scatterpolar(
                r=[current[idx]],
                theta=[labels[idx]],
                mode="markers",
                marker=dict(
                    size=22, color="rgba(0,0,0,0)",
                    line=dict(color="#f59e0b", width=3),
                ),
                name=f"Next focus: {DIM_LABELS[target_dimension]}",
                hovertemplate=f"<b>Next focus</b><br>{DIM_LABELS[target_dimension]}<extra></extra>",
            ))

        # Mean score annotation in the center
        mean_score = sum(current) / len(current)
        mean_color = _score_color(mean_score / 100)
        fig.add_annotation(
            text=f"<b style='font-size:22px'>{mean_score:.0f}</b><br>"
                 f"<span style='font-size:11px;color:#64748b'>overall</span>",
            x=0.5, y=0.5, xref="paper", yref="paper",
            showarrow=False,
            font=dict(color=mean_color, size=14),
            align="center",
        )

        fig.update_layout(
            polar=dict(
                bgcolor="#f8fafc",
                radialaxis=dict(
                    visible=True, range=[0, 100],
                    tickvals=[20, 40, 60, 80, 100],
                    ticktext=["20", "40", "60", "80", "100"],
                    tickfont=dict(size=10, color="#64748b"),
                    gridcolor="#e2e8f0",
                    linecolor="#e2e8f0",
                ),
                angularaxis=dict(
                    tickfont=dict(size=13, color="#0f172a"),
                    gridcolor="#e2e8f0",
                    linecolor="#cbd5e1",
                ),
            ),
            showlegend=True,
            legend=dict(
                orientation="h", yanchor="bottom", y=-0.15,
                xanchor="center", x=0.5,
                font=dict(size=11),
            ),
            margin=dict(l=60, r=60, t=30, b=60),
            height=420,
            paper_bgcolor="white",
        )
        return fig
    except Exception as e:
        logger.warning(f"radar render failed: {e}")
        return None


def _format_reward_table(reward_history: list) -> list:
    rows = []
    for i, r in enumerate(reward_history):
        rows.append([
            i + 1,
            f"{r.get('total', 0):.3f}",
            f"{r.get('calibration_score', 0):.3f}",
            f"{r.get('improvement_signal', 0):.3f}",
            f"{r.get('consistency_score', 0):.3f}",
            r.get("target_dimension", ""),
            r.get("difficulty", ""),
        ])
    return rows


# ---------------------------------------------------------------------------
# Gradio handlers
# ---------------------------------------------------------------------------

def start_session(state):
    if _load_error:
        return (
            gr.update(value=f"Load error: {_load_error}", interactive=False),
            gr.update(interactive=False),
            None, [], [], "Error loading model."
        )
    obs, info = _env.reset()
    baseline = obs.skill_profile.to_dict()
    state = {
        "obs": obs,
        "reward_history": [],
        "step": 0,
        "done": False,
        "baseline": baseline,
    }
    radar = _make_radar_chart(baseline, baseline_dict=None, target_dimension=None, pristine=True)
    return (
        gr.update(value=obs.question, interactive=False),
        gr.update(value="", interactive=True, placeholder="Type your answer here..."),
        radar,
        [],
        state,
        f"Domain: **{obs.domain}** | Difficulty: **{obs.difficulty}** | Ability: **{obs.student_ability_level}**",
    )


def submit_answer(answer: str, state: dict):
    if state is None or state.get("done"):
        return (
            gr.update(value="Session ended. Click 'Start New Session' to begin again."),
            gr.update(value="", interactive=False),
            None, [], state, "", ""
        )

    obs = state["obs"]
    obs.student_answer = answer

    # Scorer evaluates
    action = _scorer.score(obs) if _scorer else {d: 0.5 for d in ["correctness", "edge_case_coverage", "complexity_analysis", "tradeoff_reasoning", "communication_clarity"]}
    action["_student_answer"] = answer

    # If the scorer returned all-0.5s with an empty rationale, the model most
    # likely produced malformed JSON and Scorer._parse_output silently fell back.
    # Make this visible instead of showing a table of zeros with no explanation.
    all_half = all(
        isinstance(action.get(d), (int, float)) and abs(action[d] - 0.5) < 1e-6
        for d in ["correctness", "edge_case_coverage", "complexity_analysis", "tradeoff_reasoning", "communication_clarity"]
    )
    if all_half and not action.get("rationale"):
        action["rationale"] = (
            "Scorer returned default scores (model output could not be parsed as JSON). "
            "This usually means the model needs more fine-tuning or the prompt needs stricter formatting."
        )

    # Env step
    new_obs, reward, terminated, truncated, info = _env.step(action)
    done = terminated or truncated

    # Update state
    reward_entry = {
        "total": reward,
        "calibration_score": info.get("calibration_score", 0),
        "improvement_signal": info.get("improvement_signal", 0),
        "consistency_score": info.get("consistency_score", 0),
        "target_dimension": info.get("target_dimension", ""),
        "difficulty": new_obs.difficulty,
    }
    state["reward_history"].append(reward_entry)
    state["obs"] = new_obs
    state["step"] += 1
    state["done"] = done

    radar = _make_radar_chart(
        new_obs.skill_profile.to_dict(),
        baseline_dict=state.get("baseline"),
        target_dimension=info.get("target_dimension"),
    )
    reward_table = _format_reward_table(state["reward_history"])

    rationale = action.get("rationale", "").strip()
    if rationale:
        rationale_text = (
            f"<div style='background:#f8fafc;border-left:3px solid #2563eb;"
            f"padding:10px 14px;margin-top:8px;color:#0f172a;font-size:13px;"
            f"line-height:1.55;border-radius:2px'>"
            f"<b style='color:#334155'>Interviewer feedback</b><br>{rationale}"
            f"</div>"
        )
    else:
        rationale_text = ""

    if done:
        next_q = "Session complete. Click 'Start New Session' to try again."
        answer_box = gr.update(value="", interactive=False)
    else:
        next_q = new_obs.question
        answer_box = gr.update(value="", interactive=True)

    status = (
        f"Step {state['step']} | Domain: **{new_obs.domain}** | "
        f"Difficulty: **{new_obs.difficulty}** | "
        f"Targeting: **{info.get('target_dimension', '')}**"
    )

    return (
        gr.update(value=next_q),
        answer_box,
        radar,
        reward_table,
        state,
        status,
        rationale_text,
    )


# ---------------------------------------------------------------------------
# Build Gradio UI
# ---------------------------------------------------------------------------

def build_demo():
    with gr.Blocks(title="AdaptiveInterviewEnv Demo", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# AdaptiveInterviewEnv\n"
            "A CS technical interview environment where an RL-trained Scorer "
            "evaluates your answers across five skill dimensions."
        )

        state = gr.State(None)

        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("### Interview")
                question_box = gr.Textbox(
                    label="Question",
                    lines=3,
                    interactive=False,
                    value="Click 'Start New Session' to begin.",
                )
                answer_box = gr.Textbox(
                    label="Your Answer",
                    lines=5,
                    placeholder="Click 'Start New Session' first...",
                    interactive=False,
                )
                with gr.Row():
                    start_btn = gr.Button("Start New Session", variant="primary")
                    submit_btn = gr.Button("Submit Answer", variant="secondary")

                status_md = gr.Markdown("")
                rationale_md = gr.Markdown("")

            with gr.Column(scale=1):
                gr.Markdown("### Skill Profile")
                radar_plot = gr.Plot(label="")
                gr.Markdown(
                    "<div style='font-size:12px;color:#475569;line-height:1.6'>"
                    "Each axis is a skill the Scorer evaluates (0 to 100). "
                    "<span style='color:#16a34a;font-weight:600'>Green</span> = strong (70+), "
                    "<span style='color:#d97706;font-weight:600'>amber</span> = developing (40–69), "
                    "<span style='color:#dc2626;font-weight:600'>red</span> = weak (&lt;40). "
                    "The dotted outline marks where the session started. "
                    "The amber ring marks the dimension the Scorer will probe next."
                    "</div>"
                )

        gr.Markdown("### Reward Breakdown")
        reward_table = gr.Dataframe(
            headers=["Step", "Total", "Calibration", "Improvement", "Consistency", "Target Dim", "Difficulty"],
            datatype=["number", "str", "str", "str", "str", "str", "str"],
            interactive=False,
        )

        # Wire buttons
        start_btn.click(
            fn=start_session,
            inputs=[state],
            outputs=[question_box, answer_box, radar_plot, reward_table, state, status_md],
        )
        submit_btn.click(
            fn=submit_answer,
            inputs=[answer_box, state],
            outputs=[question_box, answer_box, radar_plot, reward_table, state, status_md, rationale_md],
        )

    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.launch(share=False)
