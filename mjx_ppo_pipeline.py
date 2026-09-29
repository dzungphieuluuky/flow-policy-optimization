import os
import functools
import jax
import jax.numpy as jnp
from brax import envs
from brax.training.agents.ppo import train as ppo
from brax.io import model

def main():
    print("=== Initializing MJX PPO Pipeline ===")
    
    # 1. Choose a pre-coded MJX environment
    env_name = "ant" 
    env = envs.get_environment(env_name=env_name, backend="mjx")
    eval_env = envs.get_environment(env_name=env_name, backend="mjx")
    
    print(f"Loaded environment: {env_name}")
    print(f"Action size: {env.action_size}")
    print(f"Observation size: {env.observation_size}")

    # 2. Configure Hyperparameters for GPU Acceleration
    train_fn = functools.partial(
        ppo.train,
        num_timesteps=30_000_000,       
        num_evals=20,                    
        reward_scaling=0.1,
        episode_length=1000,
        normalize_observations=True,
        action_repeat=1,
        unroll_length=5,
        num_minibatches=32,
        num_envs=4096,                   
        learning_rate=3e-4,
        entropy_cost=1e-2,
        discounting=0.97,
        lambda_=0.95,
        ppo_epochs=4,
        batch_size=2048,
    )

    x_data = []
    y_data = []
    
    def progress_callback(num_steps, metrics):
        reward = metrics['eval/episode_reward']
        x_data.append(num_steps)
        y_data.append(reward)
        print(f"Steps: {num_steps:,} | Eval Reward: {reward:.2f}")

    print("\nStarting training on GPU/TPU device...")
    make_inference_fn, params, _ = train_fn(
        environment=env,
        progress_fn=progress_callback,
        evaluation_environment=eval_env
    )
    print("\nTraining finished successfully!")

    os.makedirs("models", exist_ok=True)
    model_path = "models/mjx_ppo_ant_params"
    model.save_params(model_path, params)
    print(f"Policy saved to: {model_path}")

if __name__ == "__main__":
    main()
