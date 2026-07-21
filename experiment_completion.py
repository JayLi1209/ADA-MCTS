"""
Goal-completion-rate experiment for ADA-MCTS on (Non-Stationary) FrozenLake.

Unlike act_learn.py (which runs 2 seeds, one episode each, and logs only a
single final reward), this script:

  1. Prints the experimental SETTING up front -- the map, the hidden parameter
     theta (intended_prob, i.e. the probability the agent moves in its intended
     direction vs. slips), the reward structure, and which pretrained model is
     used.
  2. Runs N independent episodes, lets ADA-MCTS act in each, and classifies the
     outcome of every episode as GOAL / HOLE / TIMEOUT.
  3. Reports the GOAL COMPLETION RATE plus failure/timeout rates, mean steps and
     mean return.

Run:
    python experiment_completion.py
"""

import time
import pickle
import logging
import argparse

import autograd.numpy as np
from tqdm import tqdm

from nsfrozenlake.nsfrozenlake_v0 import NSFrozenLakeV0 as model, MAPS
from HiPMDP import HiPMDP, train_model
from adamcts import MCTS, Node
from plot_experiment_completion import plot_trial_metrics


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
DEFAULTS = dict(
    domain="frozenlake",
    map_name="4x4",
    model_itr=2,            # which pretrained BNN (MDP0) weights to load
    theta=0.7,              # intended_prob: P(move in intended direction) (used when no schedule)
    # Time-varying non-stationarity: (step, intended_prob) breakpoints. The value
    # in effect is the prob of the latest threshold <= the current step counter.
    # Here: 1.0 at step 0 (deterministic first move), then 0.7 from step 1 onward.
    intended_prob_schedule=[(0, 1.0), (1, 0.7)],
    latent_code=1,          # non-stationary time slice used at reset
    n_episodes=100,         # number of evaluation episodes
    mcts_iterations=5000,   # MCTS rollouts per decision
    max_steps=100,          # safety cap on episode length
    threshold=0.02,         # epistemic-uncertainty threshold (ADA-MCTS)
    training_started=True,  # ADA-MCTS gate active: use adapted BNN2 when more confident
    danger=False,           # pessimistic Wasserstein shaping toggle
    render=False,           # ASCII-render each step (slow)
    # ----- Online learning (adapt BNN2 on collected real transitions) -------
    online_learning=True,   # fit BNN2 online from transitions gathered while acting
    min_buffer=30,          # start training once this many transitions are buffered
    train_every=15,         # retrain after this many new transitions
)

ACTION_NAMES = {0: "Left", 1: "Down", 2: "Right", 3: "Up"}


def encode_action(action):
    a = np.array([0] * 4)
    a[action] = 1
    return a


# --------------------------------------------------------------------------- #
# Setting / theta display
# --------------------------------------------------------------------------- #
def print_setting(cfg):
    grid = MAPS[cfg["map_name"]]
    n_cells = sum(len(r) for r in grid)
    n_holes = sum(r.count("H") for r in grid)
    slip = (1.0 - cfg["theta"])

    print("=" * 66)
    print(" ADA-MCTS  --  FrozenLake goal-completion experiment")
    print("=" * 66)
    print(" Environment       : Non-Stationary FrozenLake ({})".format(cfg["map_name"]))
    print(" Map layout        :")
    for row in grid:
        print("        " + " ".join(row))
    print("        (S=start  F=frozen/safe  H=hole  G=goal)")
    print(" Grid cells        : {}  ({} holes)".format(n_cells, n_holes))
    print("-" * 66)
    print(" THETA (hidden dynamics parameter)")
    schedule = cfg.get("intended_prob_schedule")
    if schedule:
        print("   intended_prob   : time-varying (non-stationary)")
        ordered = sorted(schedule, key=lambda x: x[0])
        for i, (t, p) in enumerate(ordered):
            upto = ("step >= {}".format(t) if i == len(ordered) - 1
                    else "steps {}-{}".format(t, ordered[i + 1][0] - 1))
            print("       {:<12}: intended_prob {:.2f} | slip_prob {:.2f}"
                  .format(upto, p, 1.0 - p))
    else:
        print("   intended_prob   : {:.2f}  -> P(agent moves in intended direction)"
              .format(cfg["theta"]))
        print("   slip_prob       : {:.2f}  -> split over the perpendicular cells"
              .format(slip))
    print("   latent_code     : {}     -> non-stationary time slice at reset"
          .format(cfg["latent_code"]))
    print("-" * 66)
    print(" Reward structure  : +1 reach Goal | -1 fall in Hole | 0 otherwise")
    print(" Goal 'completion' : episode whose terminal cell is G (reward +1)")
    print("-" * 66)
    print(" Planner           : ADA-MCTS")
    print("   pretrained MDP0 : models/{}_*_weights_itr_{}"
          .format(cfg["domain"], cfg["model_itr"]))
    print("   MCTS iterations : {} per decision".format(cfg["mcts_iterations"]))
    print("   training_started: {}".format(cfg["training_started"]))
    print("   danger (pessim.): {}".format(cfg["danger"]))
    print(" Episodes          : {}".format(cfg["n_episodes"]))
    print("=" * 66)


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #
def build_planner(cfg):
    """Load pretrained BNN weights and build the two HiPMDP/BNN instances once."""
    domain, itr = cfg["domain"], cfg["model_itr"]
    with open("models/{}_network_weights_itr_{}".format(domain, itr), "rb") as f:
        network_weights = pickle.load(f)
    with open("models/{}_latent_weights_itr_{}".format(domain, itr), "rb") as f:
        latent_weights = pickle.load(f)

    common = dict(
        run_type="full",
        bnn_hidden_layer_size=25,
        bnn_num_hidden_layers=3,
        bnn_network_weights=network_weights,
    )
    preset_hidden_params = [{"latent_code": cfg["latent_code"]}]

    hipmdp1 = HiPMDP(domain, preset_hidden_params, **common)
    hipmdp1._HiPMDP__initialize_BNN()
    hipmdp2 = HiPMDP(domain, preset_hidden_params, **common)
    hipmdp2._HiPMDP__initialize_BNN()

    weight_set1 = latent_weights.reshape(latent_weights.shape[1])
    return hipmdp1, hipmdp2, weight_set1


def classify_outcome(task, last_reward):
    """Return one of 'goal' / 'hole' / 'timeout'."""
    if last_reward == 1.0:
        return "goal"
    if last_reward == -1.0:
        return "hole"
    return "timeout"


# --------------------------------------------------------------------------- #
# Single episode
# --------------------------------------------------------------------------- #
def online_update(cfg, hipmdp2, online):
    """Fit BNN2 on the transitions collected so far and feed the adapted
    network weights back into the planner. Mutates `online` and `hipmdp2`."""
    domain = cfg["domain"]
    exp_list = np.reshape(online["buffer"], [-1, 5])
    buf_path = "data_buffer/{}_online_exp_buffer".format(domain)
    with open(buf_path, "wb") as f:
        pickle.dump(exp_list, f)

    net_w, latent_w, best_net, best_lat = train_model(
        "online", domain, hipmdp2, online["full_weights2"],
        online["best_net_err"], online["best_latent_err"], 0,
    )
    # Feed the adapted weights back into the planner's second (adapting) model.
    hipmdp2.network.weights = net_w
    online["weight_set2"] = latent_w
    online["full_weights2"] = latent_w.reshape(1, latent_w.shape[0])
    online["best_net_err"] = best_net
    online["best_latent_err"] = best_lat
    online["n_updates"] += 1
    online["since_last_train"] = 0
    # Predictions are cached by (state, action) at the class level; the cache is
    # now stale because BNN2's weights changed, so invalidate it.
    Node.bnn_cache = {}


def run_episode(seed, cfg, hipmdp1, hipmdp2, weight_set1, online):
    np.random.seed(seed)

    task = model(map_name=cfg["map_name"], intended_prob=cfg["theta"],
                 intended_prob_schedule=cfg.get("intended_prob_schedule"))
    task.reset(cfg["latent_code"], seed)

    last_reward = 0.0
    steps = 0
    discounted_return = 0.0
    gamma = 0.99

    while not task.is_done() and steps < cfg["max_steps"]:
        state = task.observe()
        mcts = MCTS(
            state, task.state.index,
            hipmdp1, hipmdp2, weight_set1, online["weight_set2"],
            cfg["latent_code"] - 1, task,
            cfg["threshold"], cfg["training_started"], cfg["danger"],
        )
        mcts.search(cfg["mcts_iterations"])
        best_action = mcts.best_action()
        next_state, last_reward, _, _ = task.step(best_action)
        discounted_return += (gamma ** steps) * last_reward
        steps += 1

        if online["enabled"]:
            # transition tuple matches train_model's buffer format:
            # [state, one-hot action, reward, next_state, instance_index]
            online["buffer"].append(
                np.reshape(np.array(
                    [state, encode_action(best_action), last_reward, next_state, 0],
                    dtype=object), [1, 5]))
            online["since_last_train"] += 1
            if (len(online["buffer"]) >= cfg["min_buffer"]
                    and online["since_last_train"] >= cfg["train_every"]):
                online_update(cfg, hipmdp2, online)

        if cfg["render"]:
            task.render()
            time.sleep(0.05)

    outcome = classify_outcome(task, last_reward)
    return dict(seed=seed, outcome=outcome, steps=steps, ret=discounted_return)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(cfg):
    print_setting(cfg)

    hipmdp1, hipmdp2, weight_set1 = build_planner(cfg)

    # Online-learning state. The buffer, adapted BNN2 weights, and latent weight
    # set persist ACROSS episodes so BNN2 keeps adapting to the (non-stationary)
    # dynamics it experiences while acting.
    init_w2 = np.random.default_rng(0).normal(0.0, 0.1, (1, 5))
    online = dict(
        enabled=cfg["online_learning"],
        buffer=[],
        weight_set2=init_w2.reshape(5),
        full_weights2=init_w2,
        best_net_err=100.0,
        best_latent_err=100.0,
        since_last_train=0,
        n_updates=0,
    )
    if online["enabled"]:
        print("\n Online learning: ENABLED -- BNN2 adapts via train_model "
              "(min_buffer={}, train_every={})".format(cfg["min_buffer"], cfg["train_every"]))
    else:
        print("\n Online learning: DISABLED -- planning with frozen pretrained weights")

    results = []
    print("\n Running episodes...")
    pbar = tqdm(range(cfg["n_episodes"]), desc="episodes", unit="ep")
    for i in pbar:
        seed = 1000 + i
        # reset accumulating class-level uncertainty buffers between episodes
        Node.basemodel_epistemic = []
        Node.newmodel_epistemic = []
        t0 = time.time()
        r = run_episode(seed, cfg, hipmdp1, hipmdp2, weight_set1, online)
        r["secs"] = time.time() - t0
        results.append(r)
        n_goal_so_far = sum(x["outcome"] == "goal" for x in results)
        pbar.set_postfix(last=r["outcome"].upper(),
                         goal_rate="{:.0%}".format(n_goal_so_far / len(results)),
                         bnn_updates=online["n_updates"],
                         buf=len(online["buffer"]))
        tqdm.write(
            "   ep {:>3} | seed {:>4} | {:<8} | steps {:>3} | return {:+.3f} | "
            "bnn_updates {:>3} | {:.1f}s"
            .format(i + 1, seed, r["outcome"].upper(), r["steps"], r["ret"],
                    online["n_updates"], r["secs"]))
    pbar.close()

    n = len(results)
    n_goal = sum(r["outcome"] == "goal" for r in results)
    n_hole = sum(r["outcome"] == "hole" for r in results)
    n_to = sum(r["outcome"] == "timeout" for r in results)
    mean_steps = np.mean([r["steps"] for r in results])
    mean_ret = np.mean([r["ret"] for r in results])

    print("\n" + "=" * 66)
    print(" RESULTS  (theta = intended_prob = {:.2f}, {} episodes)"
          .format(cfg["theta"], n))
    print("-" * 66)
    print("   Goal completion rate : {:>6.1%}   ({}/{})".format(n_goal / n, n_goal, n))
    print("   Hole / failure rate  : {:>6.1%}   ({}/{})".format(n_hole / n, n_hole, n))
    print("   Timeout rate         : {:>6.1%}   ({}/{})".format(n_to / n, n_to, n))
    print("   Mean steps/episode   : {:>6.1f}".format(mean_steps))
    print("   Mean discounted ret. : {:>+6.3f}".format(mean_ret))
    print("   Online BNN updates   : {:>6d}   (online_learning={})"
          .format(online["n_updates"], online["enabled"]))
    print("=" * 66)

    logging.basicConfig(
        filename="results.log", filemode="a", level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logging.getLogger(__name__).info(
        "FrozenLake completion | theta=%.2f | episodes=%d | goal_rate=%.3f "
        "| hole_rate=%.3f | timeout_rate=%.3f | mean_steps=%.1f | mean_return=%.3f",
        cfg["theta"], n, n_goal / n, n_hole / n, n_to / n, mean_steps, mean_ret,
    )

    # Side-by-side running-average plot of per-episode metrics.
    steps_list = [r["steps"] for r in results]
    returns_list = [r["ret"] for r in results]
    goals_list = [1 if r["outcome"] == "goal" else 0 for r in results]
    schedule = cfg.get("intended_prob_schedule")
    schedule_label = ("schedule=" + str(schedule)) if schedule \
        else "theta={:.2f}".format(cfg["theta"])
    save_path = "results/frozenlake_completion_{}ep.png".format(n)
    plot_trial_metrics(steps_list, returns_list, goals_list,
                       save_path, schedule_label=schedule_label)
    print(" Saved plot to: {}".format(save_path))

    return n_goal / n


def parse_args():
    p = argparse.ArgumentParser(description="ADA-MCTS FrozenLake completion-rate experiment")
    p.add_argument("--theta", type=float, default=DEFAULTS["theta"],
                   help="intended_prob (probability of moving in intended direction)")
    p.add_argument("--episodes", type=int, default=DEFAULTS["n_episodes"])
    p.add_argument("--iterations", type=int, default=DEFAULTS["mcts_iterations"])
    p.add_argument("--model-itr", type=int, default=DEFAULTS["model_itr"])
    p.add_argument("--render", action="store_true")
    p.add_argument("--no-schedule", action="store_true",
                   help="disable the time-varying intended_prob schedule (use fixed --theta)")
    p.add_argument("--no-online", action="store_true",
                   help="disable online learning (plan with frozen pretrained weights)")
    p.add_argument("--min-buffer", type=int, default=DEFAULTS["min_buffer"],
                   help="start online training once this many transitions are buffered")
    p.add_argument("--train-every", type=int, default=DEFAULTS["train_every"],
                   help="retrain BNN2 after this many new transitions")
    args = p.parse_args()

    cfg = dict(DEFAULTS)
    cfg.update(theta=args.theta, n_episodes=args.episodes,
               mcts_iterations=args.iterations, model_itr=args.model_itr,
               render=args.render, min_buffer=args.min_buffer,
               train_every=args.train_every)
    if args.no_schedule:
        cfg["intended_prob_schedule"] = None
    if args.no_online:
        cfg["online_learning"] = False
    return cfg


if __name__ == "__main__":
    main(parse_args())
