"""Facilitator plotting helpers and concept diagrams.

Every function returns a matplotlib ``Figure`` built without pyplot, so the
helpers work the same in Jupyter, Colab, VS Code, and headless test runs. The
concept diagrams use matplotlib only, which keeps the participant environment
free of Graphviz or other system dependencies.
"""

from __future__ import annotations

import io

import numpy as np
from matplotlib.figure import Figure as _MplFigure
from matplotlib.patches import FancyBboxPatch


class Figure(_MplFigure):
    """A Figure that renders itself as the last expression of a notebook cell.

    Without pyplot, IPython never registers matplotlib's image formatter, so a
    bare Figure would display as ``<Figure size ...>`` text.
    """

    def _repr_png_(self) -> bytes:
        buffer = io.BytesIO()
        self.savefig(buffer, format="png", dpi=100, bbox_inches="tight")
        return buffer.getvalue()

BLUE = "#4472C4"
ORANGE = "#ED7D31"
GREY = "#A5A5A5"
GREEN = "#70AD47"
RED = "#C0504D"
PURPLE = "#8064A2"
DARK = "#404040"


def plot_uplift_distribution(uplift: np.ndarray) -> Figure:
    fig = Figure(figsize=(6, 4))
    ax = fig.subplots()
    ax.hist(uplift, bins=40, color=BLUE, edgecolor="white")
    ax.axvline(0.0, color="black", linewidth=1)
    ax.set_xlabel("estimated uplift (mu1 - mu0)")
    ax.set_ylabel("customers")
    ax.set_title("Estimated coupon uplift distribution")
    return fig


def plot_policy_comparison(labels: list[str], values: list[float]) -> Figure:
    fig = Figure(figsize=(6, 4))
    ax = fig.subplots()
    colors = [GREEN if value >= 0 else RED for value in values]
    bars = ax.bar(labels, values, color=colors, width=0.6)
    ax.bar_label(bars, labels=[f"{value:,.0f}" for value in values], padding=3)
    ax.axhline(0.0, color="black", linewidth=1)
    ax.margins(y=0.15)
    ax.set_ylabel("true expected value captured")
    ax.set_title("Targeting policy comparison (same budget)")
    return fig


def _draw_qini(ax, curves: dict) -> None:
    from src.evaluation import qini_coefficient

    palette = [ORANGE, BLUE, GREY, PURPLE, GREEN, RED]
    for (label, curve), color in zip(curves.items(), palette):
        ax.plot(curve["share"], curve["gain"], color=color, linewidth=2.2,
                label=f"{label}  (Qini {qini_coefficient(curve):.1f})")
    last = next(iter(curves.values()))
    ax.plot([0, 1], [0, last["gain"].iloc[-1]], color="black", linestyle="--",
            linewidth=1, label="random line")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("share of pilot customers targeted (highest score first)")
    ax.set_ylabel("incremental purchases")
    ax.legend(fontsize=9)


def plot_qini_curves(curves: dict) -> Figure:
    """Qini curves from ``{label: DataFrame[share, gain]}`` on randomized pilot data."""
    fig = Figure(figsize=(7.5, 4.8))
    ax = fig.subplots()
    _draw_qini(ax, curves)
    ax.set_title("Which ranking finds the customers the coupon moves?")
    fig.tight_layout()
    return fig


# --- Results board -----------------------------------------------------------
# The participant's closing charts. Each one uses only the participant's own
# estimates; the facilitator's complete analysis adds the simulator truth.


def _tidy(ax, title: str) -> None:
    ax.set_title(title, fontweight="bold")
    ax.grid(True, color="#E6E6E6", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def plot_effect_summary(naive: float, uplift: np.ndarray) -> Figure:
    """The naive gap next to the adjusted average effect, and the spread of uplift."""
    ate = float(np.mean(uplift))
    fig = Figure(figsize=(13, 4.6))
    left, right = fig.subplots(1, 2, gridspec_kw={"width_ratios": [0.75, 1.25]})
    bars = left.bar(["Naive observed\ndifference", "T-Learner\n(adjusted)"], [naive, ate],
                    color=[GREY, BLUE], width=0.55)
    left.bar_label(bars, labels=[f"{value:.1%}" for value in (naive, ate)], padding=4)
    left.set_ylim(0, max(naive, ate) * 1.25)
    left.set_ylabel("average coupon effect on purchase")
    _tidy(left, "Adjusting for X shrinks the gap")

    right.hist(uplift, bins=42, color=BLUE, alpha=0.88, edgecolor="white")
    right.axvline(0, color="black", linewidth=1.2, linestyle="--")
    right.axvline(ate, color=ORANGE, linewidth=2, label=f"mean {ate:.1%}")
    right.set_xlabel("estimated uplift (mu1 - mu0)")
    right.set_ylabel("customers")
    right.legend()
    _tidy(right, "The coupon moves some customers far more than others")
    fig.tight_layout()
    return fig


def plot_model_check(auc: dict, curves: dict) -> Figure:
    """Held-out AUC per model (``{name: (auc_mu0, auc_mu1)}``) beside the pilot Qini curves."""
    fig = Figure(figsize=(15, 5.2))
    left, right = fig.subplots(1, 2, gridspec_kw={"width_ratios": [0.8, 1.2]})
    names = list(auc)
    positions = np.arange(len(names))
    mu0_auc = [auc[name][0] for name in names]
    mu1_auc = [auc[name][1] for name in names]
    left.bar(positions - 0.18, mu0_auc, width=0.36, color=GREY, label="mu0 (no coupon)")
    left.bar(positions + 0.18, mu1_auc, width=0.36, color=BLUE, label="mu1 (coupon)")
    left.set_xticks(positions, names)
    left.set_ylim(0.5, max(mu0_auc + mu1_auc) + 0.05)
    left.set_ylabel("AUC on the 30% held-out customers")
    left.legend()
    _tidy(left, "Prediction check: held-out AUC")

    _draw_qini(right, curves)
    _tidy(right, "Uplift check on the randomized pilot")
    fig.tight_layout()
    return fig


def plot_budget_curve(
    profit: np.ndarray, cost: np.ndarray, purchase_probability: np.ndarray, budget: float
) -> Figure:
    """Cumulative estimated profit against spend, ranked by profit vs. by purchase probability."""
    profit, cost = np.asarray(profit), np.asarray(cost)
    fig = Figure(figsize=(9.5, 5.4))
    ax = fig.subplots()
    rankings = [
        ("Causal ranking (highest profit first)", np.argsort(-profit), ORANGE),
        ("Old AI ranking (highest purchase probability first)",
         np.argsort(-np.asarray(purchase_probability)), GREY),
    ]
    for label, order, color in rankings:
        spend, value = np.cumsum(cost[order]), np.cumsum(profit[order])
        at_budget = float(np.interp(budget, spend, value))
        ax.plot(spend, value, color=color, linewidth=2.4, label=label)
        ax.plot([budget], [at_budget], "o", color=color)
        ax.annotate(f"{at_budget:,.0f}", (budget, at_budget), xytext=(10, 12),
                    textcoords="offset points", va="center", color=color, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none"))
    ax.axvline(budget, color="black", linestyle=":", linewidth=1.5)
    ax.text(budget, 0.03, f"  Growth Budget = {budget:,.0f}", fontsize=10, va="bottom",
            transform=ax.get_xaxis_transform())
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xlabel("cumulative expected coupon spend")
    ax.set_ylabel("cumulative estimated incremental profit")
    ax.legend(loc="upper right")
    _tidy(ax, "Profit peaks, then falls: stop at positive value and the budget")
    fig.tight_layout()
    return fig


def plot_targeting_map(
    purchase_probability: np.ndarray, uplift: np.ndarray, recommended: np.ndarray
) -> Figure:
    """Every customer by purchase probability and uplift; the recommended list highlighted."""
    x, y = np.asarray(purchase_probability), np.asarray(uplift)
    chosen = np.asarray(recommended).astype(bool)
    fig = Figure(figsize=(8.4, 6.1))
    ax = fig.subplots()
    ax.scatter(x[~chosen], y[~chosen], s=9, alpha=0.18, color=GREY,
               label="Not recommended", edgecolors="none")
    ax.scatter(x[chosen], y[chosen], s=18, alpha=0.72, color=ORANGE,
               label=f"Recommended ({chosen.sum():,})", edgecolors="none")
    ax.axhline(0, color="black", linewidth=1, linestyle="--")
    ax.set_xlabel("predicted purchase probability")
    ax.set_ylabel("estimated coupon uplift")
    ax.legend(frameon=True)
    _tidy(ax, "High purchase probability is not the same as high uplift")
    fig.tight_layout()
    return fig


# --- Concept diagrams ------------------------------------------------------


def _canvas(width: float, height: float, title: str) -> tuple[Figure, object]:
    """One inch per data unit, so boxes keep their proportions."""
    fig = Figure(figsize=(width, height))
    ax = fig.subplots()
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.9)
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.axis("off")
    ax.set_title(title, fontsize=13, fontweight="bold", loc="left")
    return fig, ax


def _node(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    body: str = "",
    *,
    color: str,
    text_color: str = "white",
    edge: str | None = None,
    dashed: bool = False,
) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x - w / 2, y - h / 2),
            w,
            h,
            boxstyle="round,pad=0,rounding_size=0.15",
            facecolor=color,
            edgecolor=edge or "none",
            linewidth=1.8,
            linestyle="--" if dashed else "-",
        )
    )
    if body:
        ax.text(x, y + h * 0.2, title, ha="center", va="center",
                fontsize=11, fontweight="bold", color=text_color)
        ax.text(x, y - h * 0.17, body, ha="center", va="center",
                fontsize=9, color=text_color, linespacing=1.35)
    else:
        ax.text(x, y, title, ha="center", va="center",
                fontsize=11, fontweight="bold", color=text_color)


def _link(
    ax,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = DARK,
    width: float = 1.6,
    dashed: bool = False,
    rad: float = 0.0,
    label: str = "",
    label_xy: tuple[float, float] | None = None,
) -> None:
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(
            arrowstyle="-|>",
            color=color,
            lw=width,
            linestyle="--" if dashed else "-",
            shrinkA=0,
            shrinkB=0,
            mutation_scale=16,
            connectionstyle=f"arc3,rad={rad}",
        ),
    )
    if label:
        lx, ly = label_xy or ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
        ax.text(lx, ly, label, ha="center", va="center", fontsize=8.5,
                color=color, style="italic", linespacing=1.25,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none"))


def plot_workshop_roadmap() -> Figure:
    """Prediction -> Causality -> Policy -> Language, mapped to the tasks."""
    fig, ax = _canvas(12.2, 3.4, "Workshop roadmap: from a score to a decision")
    stages = [
        ("Prediction", "Who is likely\nto purchase?", GREY,
         "purchase_probability\n(facilitator baseline)"),
        ("Causality", "What does the\ncoupon change?", BLUE,
         "Task 1 DAG · Task 2 naive gap\nTask 3 T-Learner · Task 4 Qini"),
        ("Policy", "Who gets a coupon\nunder the budget?", ORANGE,
         "Task 5 incremental profit\n+ 2,500 Growth Budget"),
        ("Language", "Why, and what\nstill needs proof?", GREEN,
         "Five-part\ndecision memo"),
    ]
    w, h, y = 2.5, 1.7, 1.95
    xs = [1.6, 4.6, 7.6, 10.6]
    for x, (title, body, color, caption) in zip(xs, stages):
        _node(ax, x, y, w, h, title, body, color=color)
        ax.text(x, 0.5, caption, ha="center", va="center", fontsize=9,
                color=DARK, linespacing=1.3)
    for left, right in zip(xs, xs[1:]):
        _link(ax, (left + w / 2 + 0.05, y), (right - w / 2 - 0.05, y), width=2.2)
    return fig


def plot_uplift_archetypes() -> Figure:
    """The four response types as a 2x2 of potential outcomes (conceptual)."""
    fig, ax = _canvas(7.8, 6.3, "Four coupon-response archetypes")
    cells = [
        # (column, row, title, body, color)
        (0, 1, "Persuadables", "buy only because of the coupon\n→ TARGET these", GREEN),
        (1, 1, "Likely purchasers", "buy anyway; coupon is wasted cost\n← old model's favourites", GREY),
        (0, 0, "Unlikely purchasers", "no purchase either way\n→ skip", "#D9D9D9"),
        (1, 0, "Negative responders", "coupon backfires\n→ never target", RED),
    ]
    xs, ys, w, h = [2.65, 5.85], [1.75, 4.2], 3.0, 2.2
    for column, row, title, body, color in cells:
        text_color = DARK if color == "#D9D9D9" else "white"
        _node(ax, xs[column], ys[row], w, h, title, body,
              color=color, text_color=text_color)
    ax.text(xs[0], 5.62, "Would NOT buy\nwithout a coupon", ha="center",
            va="center", fontsize=10, fontweight="bold", color=DARK)
    ax.text(xs[1], 5.62, "Would buy\nwithout a coupon", ha="center",
            va="center", fontsize=10, fontweight="bold", color=DARK)
    ax.text(0.65, ys[1], "Buys WITH\na coupon", ha="center", va="center",
            rotation=90, fontsize=10, fontweight="bold", color=DARK)
    ax.text(0.65, ys[0], "Does not buy\nWITH a coupon", ha="center", va="center",
            rotation=90, fontsize=10, fontweight="bold", color=DARK)
    ax.text(0.2, 0.3, "Each customer shows only one column in real data. "
            "The archetypes are probability profiles, not observed labels.",
            fontsize=8.5, style="italic", color=DARK)
    return fig


def plot_causal_dag() -> Figure:
    """Teaching DAG: confounding path, target effect, and a forbidden control."""
    fig, ax = _canvas(11.4, 5.6, "Teaching DAG: what to adjust for, and what not to")
    _node(ax, 2.0, 2.7, 3.4, 1.7, "Customer history  X",
          "loyalty_score · orders_30d\napp_sessions_30d · recency\naverage_order_value",
          color=BLUE)
    _node(ax, 6.0, 4.3, 2.5, 0.9, "coupon_sent  (T)", color=ORANGE)
    _node(ax, 9.7, 2.7, 2.6, 0.9, "purchased_7d  (Y)", color=GREEN)
    _node(ax, 6.0, 0.95, 3.5, 1.1, "Response traits",
          "price_sensitivity · offer_affinity", color=PURPLE)
    _node(ax, 9.7, 4.75, 2.6, 0.95, "After the coupon",
          "redemption · post-coupon browsing", color="white",
          text_color=RED, edge=RED, dashed=True)

    _link(ax, (3.2, 3.55), (4.75, 4.25), label="old model targeted\nlikely buyers",
          label_xy=(3.55, 4.35))
    _link(ax, (3.7, 2.7), (8.4, 2.7), label="baseline purchase habit")
    _link(ax, (7.25, 4.05), (9.1, 3.15), color=ORANGE, width=3.0,
          label="causal effect\nwe want (uplift)", label_xy=(8.55, 3.85))
    _link(ax, (7.75, 1.2), (9.2, 2.25), color=PURPLE, dashed=True,
          label="changes how much\nthe coupon helps", label_xy=(9.35, 1.45))
    _link(ax, (7.25, 4.5), (8.4, 4.7), color=RED, dashed=True)
    _link(ax, (9.7, 4.27), (9.7, 3.15), color=RED, dashed=True)
    ax.text(9.7, 5.4, "Do NOT control for", ha="center", fontsize=9,
            fontweight="bold", color=RED)
    ax.text(0.3, 5.0, "Backdoor path  T ← X → Y  biases the naive gap.\n"
            "Conditioning on pre-treatment X closes it (if nothing is missing).",
            fontsize=9, color=DARK, linespacing=1.4, va="center")
    ax.text(0.3, 0.3, "A DAG is an identification hypothesis, not discovered truth.",
            fontsize=8.5, style="italic", color=DARK)
    return fig


def plot_t_learner_schematic() -> Figure:
    """Split by arm, fit one model per arm, predict both arms for everyone."""
    fig, ax = _canvas(12.4, 4.7, "T-Learner: two outcome models, one contrast")
    _node(ax, 1.3, 2.45, 2.2, 1.3, "observed_data", "10,000 customers\nX, T, Y",
          color=GREY)
    _node(ax, 4.2, 3.6, 2.4, 1.05, "coupon_sent = 1", "treated rows", color=ORANGE)
    _node(ax, 4.2, 1.3, 2.4, 1.05, "coupon_sent = 0", "control rows", color=BLUE)
    _node(ax, 7.2, 3.6, 2.5, 1.15, r"Model $\mu_1(x)$",
          "LogisticRegression\nfit on treated only", color=ORANGE)
    _node(ax, 7.2, 1.3, 2.5, 1.15, r"Model $\mu_0(x)$",
          "LogisticRegression\nfit on control only", color=BLUE)
    _node(ax, 10.75, 2.45, 2.9, 1.6, "Predict for EVERY row",
          r"uplift $= \mu_1(x) - \mu_0(x)$" "\n= estimated CATE", color=GREEN)

    _link(ax, (2.4, 2.8), (3.0, 3.35))
    _link(ax, (2.4, 2.1), (3.0, 1.55))
    _link(ax, (5.4, 3.6), (5.95, 3.6))
    _link(ax, (5.4, 1.3), (5.95, 1.3))
    _link(ax, (8.45, 3.45), (9.3, 2.95), color=ORANGE)
    _link(ax, (8.45, 1.45), (9.3, 1.95), color=BLUE)
    ax.text(0.2, 0.25, "Each customer is observed in one arm only; the other "
            "arm's model fills in the missing potential outcome under "
            "exchangeability, overlap and consistency.",
            fontsize=8.5, style="italic", color=DARK)
    return fig
