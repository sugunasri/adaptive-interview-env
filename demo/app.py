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
        from adaptive_interview_env.models import RewardWeights
        from adaptive_interview_env.question_generator import QuestionBank, generate_question

        # Load scorer
        model_path = HUB_REPO or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "outputs", "scorer_final",
        )
        _scorer = Scorer(model_name_or_path=model_path)

        # Build env (no student — human plays student role)
        qb_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "adaptive_interview_env", "data", "question_bank.json",
        )
        fallback_bank = QuestionBank.from_json(qb_path) if os.path.exists(qb_path) else None

        def question_gen_fn(**kwargs):
            return generate_question(fallback_bank=fallback_bank, **kwargs)

        weights = RewardWeights(calibration=0.5, improvement=0.3, consistency=0.2)
        reward_fn = RewardFunction(weights=weights, calibration_refs=[])

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

def _make_radar_chart(skill_profile_dict: dict):
    try:
        import plotly.graph_objects as go
        dims = list(skill_profile_dict.keys())
        vals = [skill_profile_dict[d] for d in dims]
        vals_closed = vals + [vals[0]]
        dims_closed = dims + [dims[0]]
        fig = go.Figure(go.Scatterpolar(
            r=vals_closed,
            theta=dims_closed,
            fill="toself",
            line_color="steelblue",
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=False,
            margin=dict(l=40, r=40, t=40, b=40),
            height=350,
        )
        return fig
    except Exception:
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
            gr.update(value=f"⚠️ Load error: {_load_error}", interactive=False),
            gr.update(interactive=False),
            None, [], [], "Error loading model."
        )
    obs, info = _env.reset()
    state = {
        "obs": obs,
        "reward_history": [],
        "step": 0,
        "done": False,
    }
    radar = _make_radar_chart(obs.skill_profile.to_dict())
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

    radar = _make_radar_chart(new_obs.skill_profile.to_dict())
    reward_table = _format_reward_table(state["reward_history"])

    rationale = action.get("rationale", "")
    rationale_text = f"**Scorer rationale:** {rationale}" if rationale else ""

    if done:
        next_q = "✅ Session complete! Click 'Start New Session' to try again."
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
        gr.Markdown("# 🎯 AdaptiveInterviewEnv\nAn RL environment that trains a **Scorer** to evaluate CS interview answers.")

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
                    start_btn = gr.Button("▶ Start New Session", variant="primary")
                    submit_btn = gr.Button("Submit Answer", variant="secondary")

                status_md = gr.Markdown("")
                rationale_md = gr.Markdown("")

            with gr.Column(scale=1):
                gr.Markdown("### Skill Profile")
                radar_plot = gr.Plot(label="Skill Radar")

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
