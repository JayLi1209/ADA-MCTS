"""ADA-MCTS-style baseline for Pendulum with MCTS + oracle dynamics.

Uses the true Pendulum simulator for rollouts (oracle baseline) and MCTS
with discretized actions. This represents the upper bound of what MCTS can
achieve on Pendulum.

Compares against our learned BNN + CEM + surprise/forget method.
"""

import math
import time
import pathlib

import numpy as np
import gymnasium as gym

_HERE = pathlib.Path(__file__).parent
LOG_FILE = str(_HERE / "pendulum_baseline_mcts.log")

# ── Config ──────────────────────────────────────────────────────────────────────
MASS_SCHEDULE = [(0, 1.0), (80, 3.0)]
CHANGE_STEPS = [80]
N_TRIALS = 20
TRIAL_LEN = 150
MCTS_ITERATIONS = 500        # MCTS simulations per action
NUM_ACTIONS = 5              # discretized torque bins
TORQUES = np.linspace(-2.0, 2.0, NUM_ACTIONS)
GAMMA = 0.99
CP = math.sqrt(2.0)
ROLLOUT_HORIZON = 10


class PendulumSim:
    """Lightweight pendulum simulator for fast MCTS rollouts."""

    def __init__(self):
        self.g = 10.0
        self.max_speed = 8.0
        self.max_torque = 2.0
        self.dt = 0.05
        self.m = 1.0
        self.l = 1.0

    def step(self, state, torque):
        th, thdot = state
        torque = np.clip(torque, -self.max_torque, self.max_torque)
        newthdot = thdot + (3 * self.g / (2 * self.l) * np.sin(th)
                             + 3.0 / (self.m * self.l ** 2) * torque) * self.dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)
        newth = th + newthdot * self.dt
        # Normalize theta
        newth = ((newth + np.pi) % (2 * np.pi)) - np.pi
        cost = float(self._angle_normalize(th) ** 2 + 0.1 * (thdot ** 2) + 0.001 * (torque ** 2))
        return np.array([newth, newthdot]), -cost

    @staticmethod
    def _angle_normalize(x):
        return ((x + np.pi) % (2 * np.pi)) - np.pi

    def observe(self, state):
        th, thdot = state
        return np.array([np.cos(th), np.sin(th), thdot], dtype=np.float32)

    def reset(self, seed=0):
        rng = np.random.default_rng(seed)
        high = np.array([np.pi, 1.0])
        state = rng.uniform(low=-high, high=high)
        return state

    def set_mass(self, m):
        self.m = m


class _Node:
    __slots__ = ("state", "action", "parent", "children", "visits", "value")
    def __init__(self, state, action=None, parent=None):
        self.state = state
        self.action = action
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0.0


def rollout(sim, state, horizon=ROLLOUT_HORIZON):
    """Random rollout from state using oracle simulator."""
    total = 0.0
    disc = 1.0
    for _ in range(horizon):
        a = np.random.randint(0, NUM_ACTIONS)
        torque = TORQUES[a]
        next_state, rew = sim.step(state, torque)
        total += disc * rew
        state = next_state
        disc *= GAMMA
    return total


def uct_score(child, parent_visits):
    if child.visits == 0:
        return float("inf")
    exploit = child.value / child.visits
    explore = CP * math.sqrt(math.log(max(1, parent_visits)) / child.visits)
    return exploit + explore


def mcts_act(sim, obs_state, n_iterations=MCTS_ITERATIONS):
    """Select action using MCTS with oracle dynamics."""
    root = _Node(obs_state)

    # Pre-expand root: create one child per action.
    for a in range(NUM_ACTIONS):
        torque = TORQUES[a]
        next_s, rew = sim.step(root.state, torque)
        child = _Node(next_s, action=a, parent=root)
        child.visits = 1
        child.value = rew
        root.children.append(child)

    for _ in range(n_iterations):
        # Selection: traverse from root using UCT.
        node = root
        while node.children:
            node = max(node.children, key=lambda c: uct_score(c, node.visits))

        # Expansion: add one child per action for this node.
        for a in range(NUM_ACTIONS):
            torque = TORQUES[a]
            next_s, rew = sim.step(node.state, torque)
            child = _Node(next_s, action=a, parent=node)
            node.children.append(child)
            node = child  # use the first child for rollout

        # Simulation: rollout from the newly expanded node.
        delta = rollout(sim, node.state)

        # Backpropagation.
        c = node
        while c is not None:
            c.visits += 1
            c.value += delta
            c = c.parent
            if c is not None:
                delta *= GAMMA

    # Pick most-visited action at root.
    visits = np.zeros(NUM_ACTIONS)
    for child in root.children:
        visits[child.action] = child.visits
    return int(np.argmax(visits)), TORQUES[int(np.argmax(visits))]


def main():
    out = open(LOG_FILE, "w")
    def log(*a):
        print(*a, file=out); out.flush()
        print(*a)

    sim = PendulumSim()
    log("=" * 80)
    log(f"MCTS (oracle dynamics) baseline | mass_schedule={MASS_SCHEDULE} | trials={N_TRIALS}")
    log(f"  MCTS iterations={MCTS_ITERATIONS} | actions={NUM_ACTIONS} | "
        f"rollout_horizon={ROLLOUT_HORIZON}")
    log("=" * 80)

    returns_hist = []
    steps_hist = []

    for trial in range(N_TRIALS):
        state = sim.reset(seed=trial)
        sim.set_mass(1.0)
        total_return = 0.0
        t0 = time.time()

        for step in range(TRIAL_LEN):
            if step in CHANGE_STEPS:
                sim.set_mass(3.0)

            obs = sim.observe(state)
            a_idx, torque = mcts_act(sim, state)
            state, rew = sim.step(state, torque)
            total_return += rew

        dt = time.time() - t0
        returns_hist.append(total_return)
        steps_hist.append(TRIAL_LEN)
        log(f"TRIAL {trial+1}: return={total_return:.1f} | time={dt:.1f}s")

    log(f"\navg return={np.mean(returns_hist):.1f} +/- {np.std(returns_hist):.1f}")
    log(f"min={np.min(returns_hist):.1f} max={np.max(returns_hist):.1f}")
    log("DONE.")
    out.close()


if __name__ == "__main__":
    main()
