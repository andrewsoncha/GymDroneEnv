import gymnasium as gym
from env import Env
from model import CustomActorCriticPolicy

from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy

if __name__ == '__main__':
    env = Env('map.png')
    check_env(env)
    model = PPO(CustomActorCriticPolicy, env, verbose=1)
    model.learn(total_timesteps=200000)
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, deterministic=True)
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
