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
    def __init__(self, visionRange = 5, imgPath=''):
        self.img = drawRandomCircles((50, 50), 5, 10)
        self.visit = np.zeros_like(self.img)
        self.imgBiggerSize = np.zeros_like
        # cv2.imshow('loaded map', self.img)
        # cv2.waitKey(1)
        self.rowN = self.img.shape[0]
        self.colN = self.img.shape[1]
        self.visionRange = visionRange

    def isOutOfBounds(self, posX, posY):
        if posX < self.visionRange or posX > self.colN-self.visionRange:
            return True
        if posY < self.visionRange or posY > self.rowN-self.visionRange:
            return True
        return False

    def getImgValue(self, posX, posY):
        if self.isOutOfBounds(posX, posY):
            return None
        else:
            return -10 if self.visit[posX][posY]==255 else self.img[posX, posY]

    def visitPos(self, posX, posY):
        if self.isOutOfBounds(posX, posY):
            return None
        else:
            self.visit[posX][posY] = 255

    def getLocalView(self, posX, posY):
        if self.isOutOfBounds(posX, posY):
            return None
        else:
            return self.img[posX-self.visionRange//2:posX+self.visionRange//2+1, posY-self.visionRange//2:posY+self.visionRange//2+1] - 2*self.visit[posX-self.visionRange//2:posX+self.visionRange//2+1, posY-self.visionRange//2:posY+self.visionRange//2+1]

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
    VISION_RANGE = 11

    def _get_obs(self):
        local_map = self.map.getLocalView(self.dronePosX, self.dronePosY)
        observation = local_map
        # cv2.imshow('localMap', cv2.resize(cv2.applyColorMap(local_map, cv2.COLORMAP_JET), (300, 300)))
        # cv2.waitKey(1)
        # print('observation shape: ', observation.shape)
        return observation
        # return observation[:, :, np.newaxis] # Used to transform (colN, rowN) array to (colN, rowN, 1) shaped array

    def _get_info(self):
        local_map = self.map.getLocalView(self.dronePosX, self.dronePosY)
        imgVal = self.map.getImgValue(self.dronePosX, self.dronePosY)
        return {
                "DronePos": (self.dronePosX, self.dronePosY),
                "local_map": local_map,
                "current_val": imgVal
                }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.map = Map(visionRange = self.VISION_RANGE, imgPath = self.map_path)
        self.dronePosX = self.map.colN//2
        self.dronePosY = self.map.rowN//2

        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def __init__(self, map_path, render_mode=""):
        self.map_path = map_path
        self.render_mode = render_mode
        self.map = Map(visionRange = self.VISION_RANGE, imgPath = map_path)
        self.dronePosX = self.map.colN//2
        self.dronePosY = self.map.rowN//2
        self.action_space = gym.spaces.Discrete(5)
        self.observation_space = gym.spaces.Box(0, 255, (self.VISION_RANGE, self.VISION_RANGE), dtype=np.uint8)
        self._action_to_direction = {
                np.int64(Actions.UP.value): np.array([0, 1]),
                np.int64(Actions.DOWN.value): np.array([0, -1]),
                np.int64(Actions.LEFT.value): np.array([-1, 0]),
                np.int64(Actions.RIGHT.value): np.array([1, 0]),
                np.int64(Actions.HOVER.value): np.array([0, 0])
                }


    def getReward(self, dronePosX, dronePosY, action):
        reward = 0
        reward += self.map.getImgValue(dronePosX, dronePosY)
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
        done = self.map.isOutOfBounds(dronePosX, dronePosY)
        truncated = False

        if not done:
            if permanent:
                self.dronePosX = dronePosX
                self.dronePosY = dronePosY
                self.map.visitPos(dronePosX, dronePosY)
                #if self.render_mode == 'rgb_array':
                #    self.render()

        return observation, float(reward), done, truncated, info

    def render(self, render_mode='rgb_array'):
        colorImage = cv2.merge([self.map.img, self.map.img, self.map.img])
        zeros = np.zeros_like(self.map.visit)
        redPath = cv2.merge([zeros, zeros, self.map.visit])
        _, mask = cv2.threshold(self.map.visit, 1, 255, cv2.THRESH_BINARY)
        mask = mask/255
        maskColor = cv2.merge([mask, mask, mask])
        frame = colorImage
        for c in range(3):
            frame[:,:,c] = colorImage[:,:,c]*(1-mask) + redPath[:,:,c]*mask
        resizedFrame = cv2.resize(frame, (200, 200))
        return resizedFrame


if __name__ == '__main__':
    env = Env('map.png')
    cv2.waitKey(1000)
