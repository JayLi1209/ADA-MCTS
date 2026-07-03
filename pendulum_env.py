"""Pendulum environment matching the interface expected by ADA-MCTS.

Continuous state [cos(theta), sin(theta), theta_dot], discretized actions.
No grid encoding/decoding — states are used directly as continuous vectors.
"""

import numpy as np
import gymnasium as gym


class PendulumEnv:
    """Pendulum-v1 wrapped for ADA-MCTS compatibility.

    State: [cos(theta), sin(theta), theta_dot]  (3-dim continuous)
    Actions: discrete indices 0..num_actions-1, mapped to torque values.
    """

    def __init__(self, num_action_bins=5, mass_schedule=None, episode_length=200):
        self._env = gym.make("Pendulum-v1")
        self.num_actions = num_action_bins
        self._torques = np.linspace(-2.0, 2.0, num_action_bins)
        self.mass_schedule = sorted(mass_schedule or [], key=lambda x: x[0])
        self._base_mass = float(self._env.unwrapped.m)
        self._episode_length = episode_length
        self._step_count = 0

    def observe(self):
        """Return the current continuous observation (3-dim)."""
        return self._obs.copy()

    def reset(self, latent_code=1, seed=0):
        obs, _ = self._env.reset(seed=seed)
        self._obs = obs.astype(np.float32)
        self._step_count = 0
        self._apply_schedule_mass()
        return self._obs

    def step(self, action):
        """Take discrete action index, return (next_obs, reward, done, _)."""
        torque = self._torques[int(action)]
        self._step_count += 1
        obs, reward, term, trunc, _ = self._env.step(np.array([torque], dtype=np.float32))
        self._obs = obs.astype(np.float32)
        self._apply_schedule_mass()
        done = term or trunc or (self._step_count >= self._episode_length)
        return self._obs.copy(), float(reward), done, {}

    def is_done(self, episode_length=None):
        if episode_length is None:
            episode_length = self._episode_length
        return self._step_count >= episode_length

    def is_terminal(self, state_index=None):
        return self._step_count >= self._episode_length

    def instant_reward_byindex(self, state_or_index):
        """For Pendulum, state is continuous; compute reward from (theta, theta_dot).

        The reward is: -(theta^2 + 0.1*theta_dot^2 + 0.001*torque^2)
        We approximate using the state values only (no torque info at rest).
        """
        if isinstance(state_or_index, (int, np.integer)):
            return -10.0  # fallback for grid states
        cos_th, sin_th, th_dot = np.asarray(state_or_index).ravel()[:3]
        theta = np.arctan2(sin_th, cos_th)
        return float(-(theta ** 2 + 0.1 * th_dot ** 2))

    @property
    def state(self):
        class StateProxy:
            index = 0
        return StateProxy()

    def _apply_schedule_mass(self):
        mass = self._base_mass
        for t, m in self.mass_schedule:
            if self._step_count >= t:
                mass = m
        self._env.unwrapped.m = mass

    def __encode_state(self, state_or_index):
        """Return continuous state vector directly (no grid encoding)."""
        if isinstance(state_or_index, (int, np.integer)):
            return self._obs
        return np.asarray(state_or_index).ravel()[:3]

    def __decode_state(self, coordinates, current_state, action):
        """Direct continuous decoding — no round(), no grid snap."""
        return np.asarray(coordinates).ravel()[:3]

    def reachable_states(self, s, a):
        """For continuous state, return empty — we handle transitions via BNN."""
        if isinstance(s, (int, np.integer)):
            return np.array([])
        return np.array([])

    def distances_matrix(self, states):
        """Euclidean distances between continuous state vectors."""
        n = len(states)
        D = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = float(np.linalg.norm(np.asarray(states[i]) - np.asarray(states[j])))
                D[i, j] = d
                D[j, i] = d
        return D
