import gymnasium as gym
from env import Env
from model import CustomActorCriticPolicy

from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import plot_results
from stable_baselines3.common import results_plotter
import numpy as np

import os

import matplotlib.pyplot as plt

import cv2

env = Env('map.png')
obs, info = env.reset()

# Manually drive it in a snake/lawnmower pattern
for _ in range(500):
    action = 2  # just go RIGHT constantly
    obs, reward, done, trunc, info = env.step(np.int64(action))
    print(f"pos={info['DronePos']} reward={reward:.3f} done={done}")
    if done:
        break
