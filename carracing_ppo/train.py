"""PPO training entrypoint for Gymnasium's CarRacing using a compact state vector."""

from __future__ import annotations

import argparse
import os
from typing import List, Optional, Sequence

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from carracing.env import make_env


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
        default=None,
        help="Override CarRacing's 1000-step limit (a lap takes ~850 steps).",
    )
    parser.add_argument("--lidar-rays", type=int, default=32, help="Lidar distances in the state (8 + this inputs).")
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
        help="Path to a saved PPO zip to resume from. If missing, training starts fresh.",
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
        "lap_time_bonus": args.lap_time_bonus,
    }


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

    policy_kwargs = dict(net_arch=dict(pi=args.policy_hidden_sizes, vf=args.policy_hidden_sizes))

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
            seed=args.seed,
            verbose=1,
        )

    callback = _build_callbacks(args, eval_env)

    total_steps = args.total_timesteps
    if args.checkpoint_frequency and args.checkpoint_frequency > 0:
        print(
            f"Training for {total_steps} steps "
            f"(checkpoints every {args.checkpoint_frequency}, eval every {args.eval_frequency})."
        )
    else:
        print(f"Training for {total_steps} steps (eval every {args.eval_frequency}).")

    model.learn(total_timesteps=total_steps, callback=callback, progress_bar=True)

    latest_path = os.path.join(args.checkpoint_dir, "ppo_latest.zip")
    model.save(latest_path)
    print(f"Training complete. Latest model saved to {latest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
