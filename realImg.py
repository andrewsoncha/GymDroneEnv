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

def make_env(rank):
    def _init():
        env = Env('map.png', render_mode = 'rgb_array')
        return env
    return _init

TRAIN_TIMESTEPS =  5_000_000
if __name__ == '__main__':
    log_dir = 'log/'
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs('./tensorboard/', exist_ok=True)

    N_ENVS = 4
    #env = SubprocVecEnv([make_env(i) for i in range(N_ENVS)])
    env = Env(render_mode = 'rgb_array')

    model = PPO('MultiInputPolicy', env, verbose=1)
    ''' model = RecurrentPPO('MultiInputLstmPolicy', env, 
                         verbose=1, 
                         n_steps = 1024, 
                         batch_size=256,
                         n_epochs=10,
                         learning_rate=3e-4,
                         tensorboard_log='./tensorboard/')
    '''
    # model.learn(total_timesteps=TRAIN_TIMESTEPS)
    # model.load('drone_search500k.zip')
    model = PPO.load('drone_search_test_train.zip')
    # model.save('drone_search')

    # vec_env = model.get_env()
    obs, info = env.reset()

    rewardSum = 0
    currentLen = 0

    rewardList = []
    lenList = []

    for i in range(1000):
        currentLen += 1
        action, _state = model.predict(obs, deterministic=True)
        #print('action:',action)
        obs, reward, done, truncated, info = env.step(int(action))
        rewardSum += reward
        print('reward: ', reward)
        print('drone_pos: ', obs['drone_pos'])

        # print('done:', done)

        if done or truncated:
            print('done! rewardSum: ', rewardSum, ' currentLen: ', currentLen)
            rewardList.append(rewardSum)
            lenList.append(currentLen)
            rewardSum = 0
            currentLen = 0
            obs, info = env.reset()
        img = env.render()
        cv2.imshow('frame', img)
        cv2.imwrite(f'frame_{i:03}.jpg', img)
        cv2.waitKey(10)
    print('rewardSum: ', rewardSum)
    if len(rewardList) > 0:
        print('reward mean: {}'.format(sum(rewardList)/len(rewardList)))
        print('episode len mean: {}'.format(sum(lenList)/len(lenList)))
        
    '''
    obs, _ = env.reset()
    for t in range(3000):
        a, _ = model.predict(obs, deterministic=False)
        obs, r, done, trunc, info = env.step(int(a))
        if t % 100 == 0:
            print(t, info['drone_pos'], round(env.map.getCoverage(), 3), round(r, 3))
        if done: break
    '''

    # env.render_mode = "rgb_array"
    # mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100, deterministic=True)
    # print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

