import gymnasium as gym
from env import Env
from model import CustomActorCriticPolicy

from stable_baselines3 import A2C
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import plot_results
from stable_baselines3.common import results_plotter

import os

import matplotlib.pyplot as plt

TRAIN_TIMESTEPS = 50000
if __name__ == '__main__':
    log_dir = 'log/'
    os.makedirs(log_dir, exist_ok=True)

    env = Env('map.png', render_mode = '')
    check_env(env)
    env = Monitor(env, log_dir)

    model = A2C(CustomActorCriticPolicy, env, verbose=1)
    model.learn(total_timesteps=TRAIN_TIMESTEPS)
    # model.save('drone_search')

    # env.render_mode = "rgb_array"
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100, deterministic=True)
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    plot_results([log_dir], TRAIN_TIMESTEPS, results_plotter.X_TIMESTEPS, "PPO results")
    plt.show()
