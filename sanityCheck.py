import gymnasium as gym
from env import Env
import numpy as np

env = Env(map_seed=42)
obs, info = env.reset()

env.step(np.int64(2))  # one step right
print("visit nonzero count:", np.count_nonzero(env.map.visit))
print("paddedVisitImg sum:", np.sum(env.map.paddedVisitImg))
print("paddedVisitMask sum:", np.sum(env.map.paddedVisitMask))
print("paddedVisit nonzero count:", np.count_nonzero(env.map.paddedVisit))
print("paddedVisitMask nonzero count:", np.count_nonzero(env.map.paddedVisitMask))
print("img_sum:", env.map.img_sum)
print("visit sum:", np.sum(env.map.visit))

total_reward = 0
coverage_history = []

for step in range(200):
    # Force the drone to move right every step
    action = np.random.randint(4)
    obs, reward, done, _, info = env.step(action)
    total_reward += reward
    coverage_history.append(env.map.getCoverage())
    print(f"Step {step:3d} | pos={info['drone_pos']} | reward={reward:.4f} | coverage={env.map.getCoverage():.4f}")
    if done:
        break
'''
for step in range(500):
    action = np.int64(np.random.randint(4))  # random direction
    obs, reward, done, _, info = env.step(action)
    total_reward += reward
    coverage_history.append(env.map.getCoverage())
    print(f"Step {step:3d} | pos={info['drone_pos']} | reward={reward:.4f} | coverage={env.map.getCoverage():.4f}")
    if done:
        break


print(f"\nTotal reward: {total_reward:.2f}")
print(f"Max coverage reached: {max(coverage_history):.4f}")
print(f"Coverage still increasing at end: {coverage_history[-1] > coverage_history[-10]}")

'''
