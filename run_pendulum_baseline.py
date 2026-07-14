"""ADA-MCTS baseline for Pendulum with latent-factor BNN, frozen snapshot,
dual-phase adaptive sampling (DPAS), and pessimistic worst-case sampling.

Matches the original ADA-MCTS algorithm (Luo et al., AAMAS 2024):
  - M_{k-1}: frozen snapshot at change point
  - M_k: online-adapting model (head + latent fine-tuned)
  - DPAS: compare epistemic uncertainty, switch between regular and pessimistic
  - GPU-batched: all N actions queried in single BNN forward pass

Run:  python run_pendulum_baseline.py
"""

import copy, math, pathlib, sys, time
import numpy as np
import torch
from torch import optim

# Import from parent Adapt repo
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from config import device
from env.pendulum import build_pendulum_env
from bnn.latent_model import make_latent_bnn, NUM_LATENT_FACTORS
from bnn.gaussian_workflow import surprise_gaussian

_HERE = pathlib.Path(__file__).parent
LOG = str(_HERE / "pendulum_baseline_mcts.log")
BNN_DIR = _HERE.parent / "data" / "pendulum_ada"

MASS_SCHEDULE = [(0, 1.0), (80, 3.0)]
CHANGE_STEPS = [80]
N_TRIALS = 20
TRIAL_LEN = 100

N_ACTIONS = 7
TORQUES = np.linspace(-2.0, 2.0, N_ACTIONS, dtype=np.float32)
MCTS_SIMS = 200
ROLLOUT_H = 8
CP = math.sqrt(2.0)
GAMMA = 0.99

EPS_E = 0.01
HEAD_LR = 1e-3
FINETUNE_EVERY = 5
MIN_BUF = 8


class _Node:
    __slots__ = ("state", "action", "parent", "children", "visits", "value")
    def __init__(self, state, action=None, parent=None):
        self.state = state; self.action = action; self.parent = parent
        self.children = []; self.visits = 0; self.value = 0.0


def _epistemic(bnn, dyn, obs_arr, act_arr, n_draws=10):
    B = obs_arr.shape[0]
    obs_t = torch.as_tensor(obs_arr, dtype=torch.float32, device=device)
    act_t = torch.as_tensor(act_arr, dtype=torch.float32, device=device)
    model_in = torch.cat([obs_t, act_t], dim=-1)
    return bnn.epistemic_variance(model_in, n_draws=n_draws)


@torch.no_grad()
def batched_predict(bnn, dyn, obs_arr, actions):
    K = len(actions)
    obs_t = torch.as_tensor(obs_arr, dtype=torch.float32, device=device)
    obs_t = obs_t.unsqueeze(0).expand(K, -1)
    act_t = torch.tensor(actions, dtype=torch.float32, device=device).unsqueeze(-1)
    state = dyn.reset(obs_t)
    next_obs, rew, _, _ = dyn.sample(act_t, state, deterministic=True)
    return next_obs.cpu().numpy(), rew.squeeze(-1).cpu().numpy()


@torch.no_grad()
def batched_rollout(bnn, dyn, obs_arr):
    obs_t = torch.as_tensor(obs_arr, dtype=torch.float32, device=device).unsqueeze(0)
    state = dyn.reset(obs_t)
    total, disc = 0.0, 1.0
    for _ in range(ROLLOUT_H):
        act = np.random.randint(0, N_ACTIONS)
        act_t = torch.tensor([[TORQUES[act]]], dtype=torch.float32, device=device)
        ns, rew, _, _ = dyn.sample(act_t, state, deterministic=True)
        total += disc * float(rew.item())
        obs_t = ns; state = dyn.reset(obs_t); disc *= GAMMA
    return total


def pessimistic_next_state(bnn, dyn, obs_arr):
    next_obs_all, rews_all = batched_predict(bnn, dyn, obs_arr, TORQUES)
    worst_idx = int(np.argmin(rews_all))
    return next_obs_all[worst_idx], float(rews_all[worst_idx])


def uct(child, parent_visits):
    if child.visits == 0:
        return float("inf")
    return child.value/child.visits + CP*math.sqrt(math.log(max(1,parent_visits))/child.visits)


def mcts_act(bnn_k, dyn_k, bnn_prev, dyn_prev, obs, training_started):
    root = _Node(obs.copy())

    saved = bnn_k.num_weight_groups
    bnn_k.num_weight_groups = 1
    if bnn_prev is not None:
        bnn_prev.num_weight_groups = 1
    try:
        next_obs_all, rews_all = batched_predict(bnn_k, dyn_k, obs, TORQUES)
        for a in range(N_ACTIONS):
            child = _Node(next_obs_all[a], action=a, parent=root)
            child.visits = 1; child.value = float(rews_all[a])
            root.children.append(child)

        for _ in range(MCTS_SIMS):
            node = root
            while node.children:
                node = max(node.children, key=lambda c: uct(c, node.visits))

            # DPAS: compare epistemic uncertainty of M_k vs M_{k-1}
            if bnn_prev is not None and training_started:
                epi_k = _epistemic(bnn_k, dyn_k,
                                   node.state[None, :],
                                   TORQUES[0:1, None])
                epi_prev = _epistemic(bnn_prev, dyn_prev,
                                      node.state[None, :],
                                      TORQUES[0:1, None])
                if epi_k + EPS_E < epi_prev:
                    ns_all, rs_all = batched_predict(bnn_k, dyn_k, node.state, TORQUES)
                else:
                    ns_worst, r_worst = pessimistic_next_state(bnn_prev, dyn_prev, node.state)
                    ns_all = np.tile(ns_worst, (N_ACTIONS, 1))
                    rs_all = np.full(N_ACTIONS, r_worst)
            else:
                ns_all, rs_all = batched_predict(bnn_k, dyn_k, node.state, TORQUES)

            for a in range(N_ACTIONS):
                child = _Node(ns_all[a], action=a, parent=node)
                node.children.append(child)

            delta = batched_rollout(bnn_k, dyn_k, node.children[0].state)
            c = node.children[0]
            while c is not None:
                c.visits += 1; c.value += delta
                c = c.parent
                if c is not None: delta *= GAMMA
    finally:
        bnn_k.num_weight_groups = saved
        if bnn_prev is not None:
            bnn_prev.num_weight_groups = 1

    visits = np.zeros(N_ACTIONS)
    for child in root.children:
        visits[child.action] = child.visits
    return int(np.argmax(visits)), TORQUES[int(np.argmax(visits))]


def main():
    out = open(LOG, "w")
    def log(*a):
        print(*a, file=out); out.flush()
        print(*a)

    bnn, dyn = make_latent_bnn(3, 1, NUM_LATENT_FACTORS)
    frozen_path = BNN_DIR / "latent_bnn_frozen.pth"
    if frozen_path.exists():
        bnn.load_latent(str(BNN_DIR), "latent_bnn_frozen.pth")
        dyn.load(str(BNN_DIR))
        log(f"Loaded pretrained latent BNN (frozen trunk) from {BNN_DIR}")
    else:
        log("ERROR: Run train_ada_mcts_bnn.py from parent repo first")
        return

    env = build_pendulum_env(MASS_SCHEDULE)
    bnn.num_weight_groups = 1
    bnn_init = copy.deepcopy(bnn.state_dict())

    log("=" * 60)
    log(f"ADA-MCTS (latent={NUM_LATENT_FACTORS}) on Pendulum | N_act={N_ACTIONS} | sims={MCTS_SIMS}")
    log(f"  mass_schedule={MASS_SCHEDULE} | {N_TRIALS} trials x {TRIAL_LEN} steps")
    log("=" * 60)

    returns = []
    for trial in range(N_TRIALS):
        obs, _ = env.reset()
        bnn.load_state_dict(bnn_init)

        bnn_prev = None
        dyn_prev = None
        training_started = False
        n_post_change = 0
        n_threshold = 3
        buffer = []
        opt = optim.Adam(bnn.head_parameters(), lr=HEAD_LR)

        total_return = 0.0
        t0 = time.time()

        for step in range(TRIAL_LEN):
            if step in CHANGE_STEPS:
                bnn_prev = copy.deepcopy(bnn)
                import mbrl.models as models
                dyn_prev = models.OneDTransitionRewardModel(
                    bnn_prev, target_is_delta=True, normalize=True, learned_rewards=True
                )
                if dyn.input_normalizer is not None:
                    dyn_prev.input_normalizer = copy.deepcopy(dyn.input_normalizer)
                training_started = False
                n_post_change = 0
                buffer.clear()
                log(f"  [CHANGE] t={step}: mass={env.unwrapped.m}, "
                    f"latent={bnn.latent.data.cpu().numpy().round(3)}")

            a_idx, torque = mcts_act(bnn, dyn, bnn_prev, dyn_prev, obs, training_started)
            act_arr = np.array([torque], dtype=np.float32)
            next_obs, reward, term, trunc, info = env.step(act_arr)
            total_return += float(reward)

            if step >= CHANGE_STEPS[0]:
                n_post_change += 1
                if n_post_change >= n_threshold:
                    training_started = True
                model_in = np.concatenate([obs, act_arr]).astype(np.float32)
                target = np.concatenate([next_obs - obs, [float(reward)]]).astype(np.float32)
                buffer.append((model_in, target))
                if len(buffer) > 100:
                    buffer.pop(0)

                if training_started and len(buffer) >= MIN_BUF and n_post_change % FINETUNE_EVERY == 0:
                    batch_size = min(16, len(buffer))
                    idxs = np.random.choice(len(buffer), size=batch_size, replace=False)
                    for i in idxs:
                        mi, tg = buffer[i]
                        mi_t = torch.tensor(mi, device=device).unsqueeze(0)
                        tg_t = torch.tensor(tg, device=device).unsqueeze(0)
                        bnn.num_train_points = max(len(buffer), 1)
                        loss, _ = bnn.loss(mi_t, tg_t)
                        loss.backward()
                    opt.step(); opt.zero_grad()
                    if n_post_change % (FINETUNE_EVERY * 3) == 0:
                        log(f"  [FT] t={step}: n_buf={len(buffer)} "
                            f"latent={bnn.latent.data.cpu().numpy().round(4)}")

            obs = next_obs
            if term or trunc:
                break

        returns.append(total_return)
        log(f"TRIAL {trial+1}: return={total_return:.1f}  ({time.time()-t0:.0f}s)")

    log(f"\nRESULTS: avg={np.mean(returns):.1f} std={np.std(returns):.1f}")
    log(f"  min={np.min(returns):.1f} max={np.max(returns):.1f}")
    out.close()


if __name__ == "__main__":
    main()
