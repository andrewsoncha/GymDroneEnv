import gymnasium as gym
from env import Env

from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import plot_results
from stable_baselines3.common import results_plotter
from stable_baselines3.common.vec_env import SubprocVecEnv

import os

import matplotlib.pyplot as plt

import cv2
import numpy as np
TOTAL_TIMESTEP_CAP = 50_000_000
def make_env(rank):
    def _init():
        env = Env(mapDirPath='selected500Imgs1/', render_mode = 'rgb_array')
        env = Monitor(env)
        return env
    return _init

if __name__ == '__main__':
    log_dir = 'log/'
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs('./tensorboard/', exist_ok=True)

    N_ENVS = 32 
    # env = SubprocVecEnv([make_env(i) for i in range(N_ENVS)])
    env = Env(mapDirPath='selected500Imgs1/', render_mode = 'rgb_array')

    model = PPO('MultiInputPolicy', env, verbose=1)
    model.learn(total_timesteps=TOTAL_TIMESTEP_CAP)
    model.save('model_real_img1_trainedPPO.zip')

