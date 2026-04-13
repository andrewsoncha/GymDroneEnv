from enum import Enum
import cv2
import numpy as np
import gymnasium as gym
from random import randint

def drawRandomCircles(imageShape, circleN, maxRadius):
    image = np.zeros(imageShape, dtype=np.uint8)
    width, height = imageShape
    for i in range(circleN):
        center = (randint(0, width), randint(0, height))
        radius = randint(0, maxRadius)
        cv2.circle(image, center, radius, color=255, thickness=-1)
    return image

class Map:
    def __init__(self, imgPath=''):
        self.img = drawRandomCircles((50, 50), 10, 10)
        # cv2.imshow('loaded map', self.img)
        # cv2.waitKey(100)
        self.rowN = self.img.shape[0]
        self.colN = self.img.shape[1]

    def isOutOfBounds(self, posX, posY):
        if posX < 0 or posX > self.colN:
            return True
        if posY < 0 or posY > self.rowN:
            return True
        return False

    def getImgValue(self, posX, posY):
        if self.isOutOfBounds(posX, posY):
            return None
        else:
            return self.img[posX, posY]

    def getLocalView(self, posX, posY, visionRange = 5):
        if self.isOutOfBounds(posX, posY):
            return None
        else:
            return self.img[posX-visionRange//2:posX+visionRange//2+1, posY-visionRange//2:posY+visionRange//2+1]

class Actions(Enum):
    UP = 0
    DOWN = 1
    RIGHT = 2
    LEFT = 3
    HOVER = 4
    
class Env(gym.Env):
    VISIT_PENALTY = -999
    HOVER_PENALTY = -1
    OUT_OF_BOUNDS_PENALTY = -9999
    VISION_RANGE = 9

    def _get_obs(self):
        local_map = self.map.getLocalView(self.dronePosX, self.dronePosY, self.VISION_RANGE)
        observation = local_map
        return observation

    def _get_info(self):
        local_map = self.map.getLocalView(self.dronePosX, self.dronePosY, self.VISION_RANGE)
        imgVal = self.map.getImgValue(self.dronePosX, self.dronePosY)
        return {
                "DronePos": (self.dronePosX, self.dronePosY),
                "local_map": local_map,
                "current_val": imgVal
                }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.map = Map(self.map_path)
        self.dronePosX = self.map.colN//2
        self.dronePosY = self.map.rowN//2
        self.visitSet = set()

        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def __init__(self, map_path):
        self.map_path = map_path
        self.map = Map(map_path)
        self.dronePosX = self.map.colN//2
        self.dronePosY = self.map.rowN//2
        self.visitSet = set()
        self.action_space = gym.spaces.Discrete(5)
        self.observation_space = gym.spaces.Box(0, 255, (self.VISION_RANGE, self.VISION_RANGE), np.uint8)
        self._action_to_direction = {
                np.int64(Actions.UP.value): np.array([0, 1]),
                np.int64(Actions.DOWN.value): np.array([0, -1]),
                np.int64(Actions.LEFT.value): np.array([-1, 0]),
                np.int64(Actions.RIGHT.value): np.array([1, 0]),
                np.int64(Actions.HOVER.value): np.array([0, 0])
                }


    def getReward(self, dronePosX, dronePosY, action):
        reward = 0
        if (dronePosX, dronePosY) in self.visitSet:
            reward += self.VISIT_PENALTY
        else:
            reward += self.map.getImgValue(dronePosX, dronePosY).mean()
        if action == 5: 
            reward += self.HOVER_PENALTY
        return reward

    def step(self, action, permanent = True):
        direction = self._action_to_direction[action]

        dronePosX = self.dronePosX + direction[0]
        dronePosY = self.dronePosY + direction[1]
        
        observation = self._get_obs()
        info = self._get_info()
        
        reward = self.getReward(self.dronePosX, self.dronePosY, action)
        done = True
        truncated = False

        if permanent:
            self.dronePosX = dronePosX
            self.dronePosY = dronePosY
            self.visitSet.add((self.dronePosX, self.dronePosY))
        return observation, float(reward), done, truncated, info


if __name__ == '__main__':
    env = Env('map.png')
    cv2.waitKey(1000)
