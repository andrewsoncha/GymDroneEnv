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

def make_env(rank):
    def _init():
        env = Env('map.png', render_mode = 'rgb_array')
        return env
    return _init

TRAIN_TIMESTEPS =  5_000_000
if __name__ == '__main__':
    env = Env(render_mode = 'rgb_array')

    obs, info = env.reset(seed=69676996)

    rewardSum = 0
    currentLen = 0

    rewardList = []
    lenList = []

    userInput = ' '
    while userInput != 'q':
        img = env.render()
        cv2.imshow('frame', img)
        cv2.waitKey(10)
        userInput = input('Input: ')
        action = 0 
        if userInput == 'w':
            action = 0
        elif userInput == 'a':
            action = 3
        elif userInput == 's':
            action = 1 
        elif userInput == 'd':
            action = 2 
        elif userInput == 'r':
            obs, info = env.reset()
            continue
        obs, reward, done, Truncated, info = env.step(int(action))
        if done:
            obs, info = env.reset()
        rewardSum += reward
        print('reward: ', reward)
        print('drone_pos: ', obs['drone_pos'])
