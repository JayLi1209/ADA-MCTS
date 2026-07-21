"""
plot_experiment_completion.py -- side-by-side trial summary plot for
experiment_completion.py.

Called at the end of experiment_completion.main() with the per-episode metrics
it collects. Produces one figure with three panels, each showing the *running
average across episodes* (cumulative mean over episodes 1..k):

    1. averaged step count   (steps to episode end)
    2. averaged return        (discounted return of the episode)
    3. averaged goal rate      (fraction of episodes that reached the goal)

The running average is plotted vs. episode index so you can see each statistic
converge as more episodes accumulate; the faint markers are the raw per-episode
values behind each average.
"""
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")          # headless: write a PNG, no display needed
import matplotlib.pyplot as plt


def plot_trial_metrics(steps, returns, goals, save_path, schedule_label=""):
    """steps, returns, goals : per-episode lists (goals is 0/1 per episode).
    Writes a 1x3 side-by-side PNG of the cumulative average of each metric."""
    steps = np.asarray(steps, dtype=np.float64)
    returns = np.asarray(returns, dtype=np.float64)
    goals = np.asarray(goals, dtype=np.float64)
    n = len(steps)
    trials = np.arange(1, n + 1)
    csum = np.arange(1, n + 1)                      # running denominator

    avg_steps = np.cumsum(steps) / csum
    avg_returns = np.cumsum(returns) / csum
    avg_goals = np.cumsum(goals) / csum

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    panels = [
        (axes[0], avg_steps, steps, "averaged step count", "steps", "tab:blue"),
        (axes[1], avg_returns, returns, "averaged return", "return", "tab:green"),
        (axes[2], avg_goals, goals, "averaged goal rate", "goal rate", "tab:red"),
    ]
    for ax, avg, raw, title, ylabel, color in panels:
        ax.plot(trials, raw, "o", color=color, alpha=0.25, ms=4,
                label="per-episode")
        ax.plot(trials, avg, "-", color=color, lw=2,
                label="running avg")
        ax.set_title(f"{title}\n(final avg = {avg[-1]:.3f})")
        ax.set_xlabel("episode")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
    axes[2].set_ylim(-0.05, 1.05)

    suffix = f"  ({schedule_label})" if schedule_label else ""
    fig.suptitle(f"ADA-MCTS FrozenLake: episode metrics over {n} episodes{suffix}",
                 y=1.02)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return save_path
