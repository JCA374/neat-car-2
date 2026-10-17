"""PPO training entrypoint for Gymnasium's CarRacing using a compact state vector."""

from __future__ import annotations

import argparse
import csv
import os
import time
from typing import List, Optional, Sequence

import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from carracing.controls import HeldControls
from carracing.env import make_env
from carracing.rules import LAP_CHECK_PPO_STEPS, MAX_FRAMES, smooth


def _parse_hidden_sizes(sizes: str) -> List[int]:
    parts = [p.strip() for p in sizes.split(",") if p.strip()]
    hidden = []
    for part in parts:
        try:
            val = int(part)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"Invalid hidden size '{part}'") from exc
        if val <= 0:
            raise argparse.ArgumentTypeError("Hidden sizes must be positive integers.")
        hidden.append(val)
    if not hidden:
        raise argparse.ArgumentTypeError("At least one hidden layer size is required.")
    return hidden


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO on CarRacing-v2 with a compact state wrapper.")
    parser.add_argument("--total-timesteps", type=int, default=500_000, help="Total environment steps to train.")
    parser.add_argument("--num-envs", type=int, default=4, help="Parallel envs (>=1).")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Adam learning rate.")
    parser.add_argument(
        "--lr-schedule",
        choices=("constant", "linear"),
        default="constant",
        help="'linear' decays the learning rate to 0 over training (stops late-run collapse).",
    )
    parser.add_argument(
        "--target-kl",
        type=float,
        default=None,
        help="Stop an update's epochs early once approx KL exceeds this (e.g. 0.02).",
    )
    parser.add_argument("--n-steps", type=int, default=1024, help="Rollout length per env before an update.")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for PPO updates.")
    parser.add_argument("--n-epochs", type=int, default=10, help="Gradient passes per batch.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda.")
    parser.add_argument("--clip-range", type=float, default=0.2, help="PPO clip range.")
    parser.add_argument("--entropy-coef", type=float, default=0.01, help="Entropy bonus coefficient.")
    parser.add_argument("--vf-coef", type=float, default=0.5, help="Value loss coefficient.")
    parser.add_argument("--max-grad-norm", type=float, default=0.5, help="Gradient clipping norm.")
    parser.add_argument(
        "--policy-hidden-sizes",
        type=_parse_hidden_sizes,
        default=_parse_hidden_sizes("256,256"),
        help="Comma-separated MLP hidden sizes for policy/value networks.",
    )
    parser.add_argument("--domain-randomize", action="store_true", help="Enable Gymnasium domain randomization.")
    parser.add_argument(
        "--offtrack-penalty",
        type=float,
        default=1.0,
        help="Penalty scale when traction falls below traction-threshold (discourages leaving track).",
    )
    parser.add_argument(
        "--traction-threshold",
        type=float,
        default=0.95,
        help="Traction value above which no offtrack penalty is applied.",
    )
    parser.add_argument(
        "--allow-multiple-laps",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep driving after a lap (bonus + fresh tiles). Disable with --no-allow-multiple-laps.",
    )
    parser.add_argument(
        "--lap-completion-bonus",
        type=float,
        default=100.0,
        help="Bonus reward given when completing a lap (when allow-multiple-laps is enabled).",
    )
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=MAX_FRAMES,
        help="Frames per training episode (rulebook: the same 2000 as lap checks and the benchmark).",
    )
    parser.add_argument("--track-seeds", default="train",
                        help="'train': draw tracks from the rulebook's training range (never a reserved track).")
    parser.add_argument("--action-smoothing", type=float, default=0.0,
                        help="Blend each action with the previous one (rulebook: 0.1, as NEAT does).")
    parser.add_argument("--controls", choices=("raw3", "throttle", "discrete"), default="raw3",
                        help="How the policy's outputs become steer/gas/brake (carracing/controls.py).")
    parser.add_argument("--action-repeat", type=int, default=1,
                        help="Decide every N frames and hold the controls in between. --total-timesteps and "
                             "--lap-check-frequency count frames; --n-steps counts decisions.")
    parser.add_argument("--normalize-reward", action="store_true", help="VecNormalize the reward (not the observations).")
    parser.add_argument("--use-sde", action="store_true", help="generalised State-Dependent Exploration (gSDE).")
    parser.add_argument("--sde-sample-freq", type=int, default=-1, help="gSDE: resample the noise every N steps.")
    parser.add_argument("--log-std-init", type=float, default=0.0, help="Initial log std of the action noise.")
    parser.add_argument("--lidar-rays", type=int, default=32, help="Lidar distances in the state (8 + this inputs).")
    parser.add_argument("--drop-heading", action="store_true", help="Leave cos/sin(heading) out of the state.")
    parser.add_argument("--lap-check-frequency", type=int, default=LAP_CHECK_PPO_STEPS,
                        help="Every N steps: lap check on the validation tracks + checkpoint (0 = off).")
    parser.add_argument("--max-hours", type=float, default=None, help="Stop after this much training (needs lap checks).")
    parser.add_argument("--patience-hours", type=float, default=None,
                        help="Stop when the lap check has not improved for this long (needs lap checks).")
    parser.add_argument(
        "--stuck-patience",
        type=int,
        default=None,
        help="End an episode after this many frames without a new tile, charging the remaining time penalty.",
    )
    parser.add_argument(
        "--lidar-fov",
        type=float,
        default=None,
        help="Forward lidar fan in degrees (default: original 324-degree layout).",
    )
    parser.add_argument(
        "--lap-time-bonus",
        type=float,
        default=0.0,
        help="Reward per frame a lap finishes under 1000 frames (0 = off).",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed for reproducibility.")
    parser.add_argument(
        "--checkpoint-dir",
        default="outputs/ppo/dev/checkpoints",
        help="Directory for best/latest checkpoints.",
    )
    parser.add_argument("--log-dir", default="outputs/ppo/dev/logs", help="Directory for monitor/tensorboard logs.")
    parser.add_argument(
        "--eval-frequency",
        type=int,
        default=25_000,
        help="How often to run eval (env steps). Set 0 to disable.",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=3,
        help="Episodes per evaluation run.",
    )
    parser.add_argument(
        "--checkpoint-frequency",
        type=int,
        default=50_000,
        help="How often to write periodic checkpoints (env steps). Set 0 to disable.",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Path to a saved PPO zip to resume from, or 'auto': continue a paused run from its last lap-check "
             "checkpoint (training time and lap checks carry on). If missing, training starts fresh.",
    )
    return parser.parse_args(argv)


def _learning_rate(args: argparse.Namespace):
    if args.lr_schedule == "linear":
        initial = args.learning_rate
        return lambda progress_remaining: initial * progress_remaining
    return args.learning_rate


def _env_kwargs(args: argparse.Namespace) -> dict:
    return {
        "render_mode": None,
        "domain_randomize": args.domain_randomize,
        "offtrack_penalty": args.offtrack_penalty,
        "traction_threshold": args.traction_threshold,
        "allow_multiple_laps": args.allow_multiple_laps,
        "lap_completion_bonus": args.lap_completion_bonus,
        "max_episode_steps": args.max_episode_steps,
        "lidar_rays": args.lidar_rays,
        "lidar_fov": args.lidar_fov,
        "drop_heading": args.drop_heading,
        "stuck_patience": args.stuck_patience,
        "lap_time_bonus": args.lap_time_bonus,
        "track_seeds": None if args.track_seeds == "any" else args.track_seeds,
        "action_smoothing": args.action_smoothing,
        "controls": args.controls,
        "action_repeat": args.action_repeat,
    }


def _paused_run(checkpoint_dir: str, log_dir: str) -> list:
    """Lap checks of an earlier, unfinished run of this step whose checkpoint exists (else [])."""
    path = os.path.join(log_dir, "validation_laps.csv")
    if not os.path.exists(path):
        return []
    with open(path, newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if r.get("time_s")]
    while rows and not os.path.exists(os.path.join(checkpoint_dir, f"val_{rows[-1]['timestep']}.zip")):
        rows.pop()
    return rows


def _make_vector_envs(
    env_kwargs: dict,
    num_envs: int,
    seed: int,
    monitor_dir: str,
    vec_cls,
    start_method: Optional[str] = None,
):
    vec_env_kwargs = {}
    if start_method and vec_cls is SubprocVecEnv:
        vec_env_kwargs["start_method"] = start_method
    return make_vec_env(
        lambda: make_env(**env_kwargs),
        n_envs=num_envs,
        seed=seed,
        vec_env_cls=vec_cls,
        monitor_dir=monitor_dir,
        monitor_kwargs={"info_keywords": ("laps_completed",)},
        vec_env_kwargs=vec_env_kwargs or None,
    )


class _Driver:
    """Deterministic policy wrapper in the shape carracing.benchmark.drive expects."""

    def __init__(self, model, action_smoothing: float = 0.0, controls: str = "raw3", repeat: int = 1) -> None:
        self.model = model
        self.smoothing = action_smoothing
        self.held = HeldControls(controls, repeat)
        self.reset()

    def reset(self) -> None:
        self.prev = np.zeros(3, dtype=np.float32)
        self.held.reset()

    def __call__(self, obs):
        action = self.held(lambda: self.model.predict(obs, deterministic=True)[0])
        if self.smoothing:
            action = self.prev = smooth(action, self.prev, self.smoothing)
        return action


class LapCheckCallback(BaseCallback):
    """Every `frequency` steps: lap check on the 5 validation tracks, save the checkpoint,
    log logs/validation_laps.csv, and stop at the time budget or when the laps stop improving.
    The check runs inside the training clock, so it counts as training time (as in NEAT)."""

    def __init__(self, frequency: int, checkpoint_dir: str, log_dir: str, env_kwargs: dict,
                 max_hours: Optional[float] = None, patience_hours: Optional[float] = None,
                 action_smoothing: float = 0.0, controls: str = "raw3", repeat: int = 1) -> None:
        super().__init__()
        self.controls = controls
        self.repeat = max(1, repeat)
        self.frequency = frequency
        self.checkpoint_dir = checkpoint_dir
        self.csv_path = os.path.join(log_dir, "validation_laps.csv")
        self.env_kwargs = env_kwargs
        self.action_smoothing = action_smoothing
        self.max_s = max_hours * 3600 if max_hours else None
        self.patience_s = patience_hours * 3600 if patience_hours else None
        self.next_check = frequency
        self.best = None
        self.best_at = 0.0
        self.offset = 0.0  # training seconds before this process (a resumed run)

    def resume_from(self, rows: list) -> None:
        """Carry on a paused run: its lap checks so far and the training time they took."""
        self.offset = float(rows[-1]["time_s"]) if rows else 0.0
        for r in rows:
            key = (int(r["finished"]), -(float(r["mean_lap_s"]) if r["mean_lap_s"] else 1e9))
            if self.best is None or key > self.best:
                self.best, self.best_at = key, float(r["time_s"])

    def _on_training_start(self) -> None:
        self.start = time.time() - self.offset
        self.next_check = self.num_timesteps + self.frequency
        if not self.offset:
            with open(self.csv_path, "w", newline="") as fh:
                csv.writer(fh).writerow(["timestep", "time_s", "finished", "mean_lap_s"])

    def _on_step(self) -> bool:
        from carracing.benchmark import lap_check

        if self.num_timesteps < self.next_check:
            return True
        self.next_check += self.frequency
        frames = self.num_timesteps * self.repeat
        laps = lap_check(_Driver(self.model, self.action_smoothing, self.controls, self.repeat), self.env_kwargs)
        self.model.save(os.path.join(self.checkpoint_dir, f"val_{frames}.zip"))
        env = self.model.get_env()
        if isinstance(env, VecNormalize):  # the reward scale, so a paused run can resume where it was
            env.save(os.path.join(self.checkpoint_dir, "vecnormalize.pkl"))
        now = time.time() - self.start
        with open(self.csv_path, "a", newline="") as fh:
            csv.writer(fh).writerow([frames, f"{now:.1f}", laps["finished"],
                                     "" if laps["mean_lap_s"] is None else laps["mean_lap_s"]])
        print(f"Lap check at {frames} frames: {laps['finished']}/5 validation tracks"
              + (f", mean lap {laps['mean_lap_s']:.2f} s" if laps["mean_lap_s"] else ""), flush=True)
        key = (laps["finished"], -(laps["mean_lap_s"] or 1e9))
        if self.best is None or key > self.best:
            self.best, self.best_at = key, now
        if self.max_s and now >= self.max_s:
            print(f"Stopped: time budget of {self.max_s / 3600:g} h reached.", flush=True)
            return False
        if self.patience_s and now - self.best_at >= self.patience_s:
            print(f"Stopped: no better lap check for {self.patience_s / 3600:g} h.", flush=True)
            return False
        return True


def _build_callbacks(
    args: argparse.Namespace,
    eval_env,
):
    callbacks = []
    if args.checkpoint_frequency and args.checkpoint_frequency > 0:
        callbacks.append(
            CheckpointCallback(
                save_freq=max(1, args.checkpoint_frequency // max(1, args.num_envs)),
                save_path=args.checkpoint_dir,
                name_prefix="ppo_carracing",
            )
        )
    if args.eval_frequency and args.eval_frequency > 0:
        callbacks.append(
            EvalCallback(
                eval_env,
                best_model_save_path=args.checkpoint_dir,
                log_path=args.log_dir,
                eval_freq=max(1, args.eval_frequency // max(1, args.num_envs)),
                n_eval_episodes=args.eval_episodes,
                deterministic=True,
                render=False,
            )
        )
    if args.lap_check_frequency and args.lap_check_frequency > 0:
        env_kwargs = {"lidar_rays": args.lidar_rays, "lidar_fov": args.lidar_fov, "drop_heading": args.drop_heading}
        callbacks.append(LapCheckCallback(args.lap_check_frequency // args.action_repeat, args.checkpoint_dir,
                                          args.log_dir, env_kwargs, args.max_hours, args.patience_hours,
                                          args.action_smoothing, args.controls, args.action_repeat))
    if not callbacks:
        return None
    return CallbackList(callbacks)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.num_envs < 1:
        raise ValueError("--num-envs must be >= 1")
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    env_kwargs = _env_kwargs(args)
    resume_rows = []
    if args.resume == "auto":
        args.resume = None
        resume_rows = _paused_run(args.checkpoint_dir, args.log_dir)
        if resume_rows:
            args.resume = os.path.join(args.checkpoint_dir, f"val_{resume_rows[-1]['timestep']}.zip")
            with open(os.path.join(args.log_dir, "validation_laps.csv"), "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=["timestep", "time_s", "finished", "mean_lap_s"])
                w.writeheader()  # drop lap checks after the checkpoint we continue from
                w.writerows(resume_rows)
            done_s = float(resume_rows[-1]["time_s"])
            if args.max_hours and done_s >= args.max_hours * 3600:
                print(f"Time budget already used ({done_s / 3600:.2f} h); nothing to train.", flush=True)
                return 0
            # Keep the earlier Monitor CSVs: the new envs would overwrite monitor/<i>.monitor.csv.
            old = os.path.join(args.log_dir, "monitor")
            if os.path.isdir(old):
                n = 1
                while os.path.exists(f"{old}_part{n}"):
                    n += 1
                os.rename(old, f"{old}_part{n}")

    train_monitor = os.path.join(args.log_dir, "monitor")
    vec_cls = DummyVecEnv if args.num_envs == 1 else SubprocVecEnv
    start_method = "fork" if vec_cls is SubprocVecEnv else None
    try:
        train_env = _make_vector_envs(
            env_kwargs, args.num_envs, args.seed, train_monitor, vec_cls, start_method
        )
    except PermissionError as exc:
        if vec_cls is not SubprocVecEnv:
            raise
        print(f"SubprocVecEnv unavailable ({exc}); falling back to DummyVecEnv with num_envs=1.", flush=True)
        args.num_envs = 1
        train_env = _make_vector_envs(env_kwargs, 1, args.seed, train_monitor, DummyVecEnv)

    # Separate monitor dir: sharing "monitor/" made eval and train env 0 both write 0.monitor.csv.
    eval_env = _make_vector_envs(
        env_kwargs, 1, args.seed + 42, os.path.join(args.log_dir, "eval_monitor"), DummyVecEnv
    )

    stats = os.path.join(args.checkpoint_dir, "vecnormalize.pkl")
    if args.normalize_reward and args.resume and os.path.exists(stats):
        train_env = VecNormalize.load(stats, train_env)
        eval_env = VecNormalize(eval_env, norm_obs=False, norm_reward=False, training=False)
    elif args.normalize_reward:
        # Scale rewards by a running estimate of the return's spread; observations are already in [-1, 1].
        train_env = VecNormalize(train_env, norm_obs=False, norm_reward=True, gamma=args.gamma)
        eval_env = VecNormalize(eval_env, norm_obs=False, norm_reward=False, training=False)

    policy_kwargs = dict(net_arch=dict(pi=args.policy_hidden_sizes, vf=args.policy_hidden_sizes))
    if args.controls != "discrete":
        policy_kwargs["log_std_init"] = args.log_std_init

    if args.resume and os.path.exists(args.resume):
        print(f"Resuming PPO from {args.resume}")
        model = PPO.load(args.resume, env=train_env, device="auto")
    else:
        model = PPO(
            policy="MlpPolicy",
            env=train_env,
            learning_rate=_learning_rate(args),
            target_kl=args.target_kl,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            ent_coef=args.entropy_coef,
            vf_coef=args.vf_coef,
            max_grad_norm=args.max_grad_norm,
            tensorboard_log=args.log_dir,
            policy_kwargs=policy_kwargs,
            use_sde=args.use_sde,
            sde_sample_freq=args.sde_sample_freq,
            seed=args.seed,
            verbose=1,
        )

    callback = _build_callbacks(args, eval_env)
    if resume_rows and callback is not None:
        for cb in callback.callbacks:
            if isinstance(cb, LapCheckCallback):
                cb.resume_from(resume_rows)

    total_steps = args.total_timesteps // args.action_repeat  # frames -> decisions
    if args.resume:
        print(f"Continuing at {model.num_timesteps * args.action_repeat:,} frames"
              + (f", {float(resume_rows[-1]['time_s']) / 3600:.2f} h of training so far" if resume_rows else ""), flush=True)
        total_steps = max(0, total_steps - model.num_timesteps)
    if args.checkpoint_frequency and args.checkpoint_frequency > 0:
        print(
            f"Training for {total_steps} steps "
            f"(checkpoints every {args.checkpoint_frequency}, eval every {args.eval_frequency})."
        )
    else:
        print(f"Training for {total_steps} steps (eval every {args.eval_frequency}).")

    model.learn(total_timesteps=total_steps, callback=callback, progress_bar=True,
                reset_num_timesteps=not args.resume)

    latest_path = os.path.join(args.checkpoint_dir, "ppo_latest.zip")
    model.save(latest_path)
    print(f"Training complete. Latest model saved to {latest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
