import gymnasium as gym
from env import Env
from model import CustomActorCriticPolicy

from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import plot_results
from stable_baselines3.common import results_plotter
import cv2

import os

import matplotlib.pyplot as plt

TRAIN_TIMESTEPS = 5_000_000
if __name__ == '__main__':
    log_dir = 'log/'
    os.makedirs(log_dir, exist_ok=True)

    env = Env(render_mode = '')
    check_env(env)
    env = Monitor(env, log_dir)
    # env = gym.make('CartPole-v1', render_mode='rgb_array')

    # model = PPO(ActorCriticPolicy, env, verbose=1)
    model = PPO('MultiInputPolicy', env, verbose=1)
    model.learn(total_timesteps=TRAIN_TIMESTEPS)
    # model.save('drone_search')

    obs, info = env.reset()

    rewardSum = 0
    
    for i in range(1000):
        action, _state = model.predict(obs, deterministic=True)
        #print('action:',action)
        obs, reward, done, Truncated, info = env.step(int(action))
        print('reward: ', reward)
        print('info: ', info['drone_pos'])
        rewardSum += reward

        # print('done:', done)

        if done:
            print('done! rewardSum: ', rewardSum)
            rewardSum = 0
            env.reset()
        img = env.render()
        cv2.imshow('frame', img)
        cv2.imwrite(f'frame_{i:03}.jpg', img)
        cv2.waitKey(10)

