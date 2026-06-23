import gymnasium as gym
from env import Env

from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import plot_results
from stable_baselines3.common import results_plotter
from stable_baselines3.common.callbacks import BaseCallback

from stable_baselines3.common.vec_env import SubprocVecEnv

import os

import matplotlib.pyplot as plt

import cv2
import numpy as np

class StopOnPlateauCallback(BaseCallback):
    def __init__(self, check_freq=10_000, patience=5, min_delta=1.0, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.patience = patience
        self.min_delta = min_delta
        self.best_mean_reward = -np.inf
        self.no_improve_count = 0

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            # Mean reward from Monitor logs
            if len(self.model.ep_info_buffer) > 0:
                mean_reward = np.mean([ep['r'] for ep in self.model.ep_info_buffer])
                if mean_reward > self.best_mean_reward + self.min_delta:
                    self.best_mean_reward = mean_reward
                    self.no_improve_count = 0
                else:
                    self.no_improve_count += 1
                    if self.verbose:
                        print(f"No improvement for {self.no_improve_count}/{self.patience} checks")
                if self.no_improve_count >= self.patience:
                    print(f"Stopping: no improvement over {self.patience} checks")
                    return False  # stops training
        return True

def make_env(rank):
    def _init():
        env = Env('map.png', render_mode = 'rgb_array')
        return env
    return _init

TRAIN_TIMESTEPS =  5_000
if __name__ == '__main__':
    log_dir = 'log/'
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs('./tensorboard/', exist_ok=True)

    N_ENVS = 4
    #env = SubprocVecEnv([make_env(i) for i in range(N_ENVS)])
    env = Env(render_mode = 'rgb_array')
    
    plateau_callback = StopOnPlateauCallback()

    policy_kwargs = dict(
            n_lstm_layers = 1024,
            lstm_hidden_size = 128
    )

    model = PPO('MultiInputPolicy', env, verbose=1)
    '''
    model = RecurrentPPO('MultiInputLstmPolicy', env, 
                         verbose=1, 
                         batch_size=256,
                         n_epochs=10,
                         learning_rate=3e-4,
                         tensorboard_log='./tensorboard/',
                         policy_kwargs = policy_kwargs)
    '''
    model.learn(total_timesteps=TRAIN_TIMESTEPS, callback=plateau_callback)
    model.save('drone_search.zip')

    # vec_env = model.get_env()
    obs, info = env.reset()

    rewardSum = 0
    
    for i in range(1000):
        action, _state = model.predict(obs, deterministic=True)
        #print('action:',action)
        obs, reward, done, Truncated, info = env.step(int(action))
        print('drone_pos:', obs['drone_pos'])
        rewardSum += reward
        print('reward: ', reward)

        # print('done:', done)

        if done:
            print('done! rewardSum: ', rewardSum)
            rewardSum = 0
            env.reset()
        img = env.render()
        cv2.imshow('frame', img)
        cv2.imwrite(f'frame_{i:03}.jpg', img)
        cv2.waitKey(10)
        

    # env.render_mode = "rgb_array"
    # mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100, deterministic=True)
    # print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    plot_results([log_dir], TRAIN_TIMESTEPS, results_plotter.X_TIMESTEPS, "PPO results")
    plt.show()
