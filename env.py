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
        self.img = drawRandomCircles((500, 500), 60, 35)
        maxVal = np.max(self.img)
        # self.img = (cv2.distanceTransform(self.img, cv2.DIST_L2, 0)*12).astype(np.uint8)
        self.visit = np.zeros_like(self.img)
        self.imgBiggerSize = np.zeros_like
        # cv2.imshow('loaded map', self.img)
        # cv2.waitKey(1)
        self.rowN = self.img.shape[0]
        self.colN = self.img.shape[1]
        self.visionRange = visionRange

    def isOutOfBounds(self, posX, posY):
        if posX < self.visionRange//2+1 or posX > self.colN-self.visionRange//2-1:
            return True
        if posY < self.visionRange//2+1 or posY > self.rowN-self.visionRange//2-1:
            return True
        return False

    def getDistToBounds(self, posX, posY):
        leftDist = posX - self.visionRange//2
        rightDist = self.colN - self.visionRange//2 - posX
        upDist = posY - self.visionRange//2
        downDist = self.rowN - self.visionRange//2 - posY
        return min([leftDist, rightDist, upDist, downDist])

    def getImgValue(self, posX, posY):
        if self.isOutOfBounds(posX, posY):
            return None
        else:
            return self.img[posX, posY]

    def visitPos(self, posX, posY):
        self.visit[posX-self.visionRange//2:posX+self.visionRange//2+1, posY-self.visionRange//2:posY+self.visionRange//2+1] = 255

    # The method is named kinda wrong. Returns if there is any cell that has not been seen before visible currently -- Andrew Chang Apr. 22 2026
    def isVisited(self, posX, posY):
        if self.isOutOfBounds(posX, posY):
            return None

        local_map, local_visit = self.getLocalView(posX, posY)
        if 0 in local_visit:
            return True
        else:
            return False

    def getLocalView(self, posX, posY):
        if self.isOutOfBounds(posX, posY):
            mapView = [[0]*self.visionRange]*self.visionRange
            visitView = [[0]*self.visionRange]*self.visionRange
        else:
            mapView = self.img[posX-self.visionRange//2:posX+self.visionRange//2+1, posY-self.visionRange//2:posY+self.visionRange//2+1] 
            visitView = self.visit[posX-self.visionRange//2:posX+self.visionRange//2+1, posY-self.visionRange//2:posY+self.visionRange//2+1]
        return mapView, visitView

    def getTargetInView(self, dronePosX, dronePosY):
        if self.isOutOfBounds(dronePosX, dronePosY):
            return 0, 0, 0, self.visionRange**2

        local_map, local_visit = self.getLocalView(dronePosX, dronePosY)

        # Part of the localview map that is not seen previously
        newlySeenCoor = [(i, j) for i in range(len(local_visit)) for j in range(len(local_visit[i])) if local_visit[i][j]==0]
        new_cells_seen_cnt = len(newlySeenCoor)
        new_map_view = [local_map[x, y] for (x, y) in newlySeenCoor]
        new_map_uniques, new_map_counts = np.unique(new_map_view, return_counts=True)
        new_map_countDict = dict(zip(new_map_uniques, new_map_counts))

        new_nonTarget_seen = 0
        if 0 in new_map_countDict:
            new_nonTarget_seen = new_map_countDict[0]
        new_target_seen = new_cells_seen_cnt - new_nonTarget_seen

        old_cells_cnt = self.visionRange**2 - new_cells_seen_cnt

        return new_cells_seen_cnt, new_target_seen, new_nonTarget_seen, old_cells_cnt

    def getCoverage(self):
        visit_size = len(self.visit) * len(self.visit[0])
        visit_uniques, visit_counts = np.unique(self.visit, return_counts=True)
        visit_countDict = dict(zip(visit_uniques, visit_counts))
        
        seen_cnt = 0
        # Put like this to avoid KeyError when there is no 255 in visit_countDict
        if 255 in visit_countDict:
            seen_cnt = visit_countDict[255]

        coverage = seen_cnt/visit_size
        return coverage

class Actions(Enum):
    UP = 0
    DOWN = 1
    RIGHT = 2
    LEFT = 3
    HOVER = 4

class Env(gym.Env):
    VISION_RANGE = 11
    BOUNDS_MARGIN = 50

    DEFAULT_PENALTY = -0.01
    NEW_NONTARGET_REWARD = 0.05
    NEW_TARGET_REWARD = 1.0
    ALREADY_SEEN_PENALTY = -0.005
    CLOSE_TO_BOUNDS_PENALTY = -0.05
    END_COVERAGE_THRESH = 0.85
    COVERAGE_END_REWARD = 5.0
    OUT_OF_BOUNDS_PENALTY = -2.0
    VISIT_PENALTY = -0.02
    HOVER_PENALTY = -0.1
    STAY_STILL_PENALTY = -0.1

    def _get_obs(self):
        local_map, local_visit = self.map.getLocalView(self.dronePosX, self.dronePosY)
        observation = {
                'local_map': local_map,
                'local_visit': local_visit,
                'drone_pos': np.array([self.dronePosX/self.map.colN, self.dronePosY/self.map.rowN])
                }
        return observation

    def _get_info(self):
        local_map, local_visit = self.map.getLocalView(self.dronePosX, self.dronePosY)
        imgVal = self.map.getImgValue(self.dronePosX, self.dronePosY)
        return {
                "DronePos": (self.dronePosX, self.dronePosY),
                "local_map": local_map,
                "local_visit": local_visit,
                "current_val": imgVal
                }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.map = Map(visionRange = self.VISION_RANGE, imgPath = self.map_path)
        self.dronePosX = self.map.colN//2
        self.dronePosY = self.map.rowN//2
        self.stayStillCnt = 0
        self.map.visitPos(self.dronePosX, self.dronePosY)

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
        self.observation_space = gym.spaces.Dict(
                {
                    "local_map": gym.spaces.Box(low=0, high=255, shape=(self.VISION_RANGE, self.VISION_RANGE), dtype=np.uint8),
                    "local_visit": gym.spaces.Box(low=0, high=255, shape=(self.VISION_RANGE, self.VISION_RANGE), dtype=np.uint8),
                    "drone_pos": gym.spaces.Box(low=0, high=1, shape=(2, ), dtype=np.float64)
                    }
                )
        self._action_to_direction = {
                np.int64(Actions.UP.value): np.array([0, 1]),
                np.int64(Actions.DOWN.value): np.array([0, -1]),
                np.int64(Actions.LEFT.value): np.array([-1, 0]),
                np.int64(Actions.RIGHT.value): np.array([1, 0]),
                np.int64(Actions.HOVER.value): np.array([0, 0])
                }
        self.stayStillCnt = 0

    def getReward(self, dronePosX, dronePosY, action):
        reward = self.DEFAULT_PENALTY

        new_cells_seen_cnt, new_target_seen, new_nonTarget_seen, old_cells_cnt = self.map.getTargetInView(dronePosX, dronePosY)

        reward += self.NEW_TARGET_REWARD * new_target_seen

        reward += self.NEW_NONTARGET_REWARD * new_nonTarget_seen

        reward += self.ALREADY_SEEN_PENALTY * old_cells_cnt

        if action == 4: 
            reward += self.HOVER_PENALTY
        if self.stayStillCnt > 2:
            reward += self.STAY_STILL_PENALTY * (self.stayStillCnt - 2)

        reward += self.CLOSE_TO_BOUNDS_PENALTY * max(0, self.BOUNDS_MARGIN - self.map.getDistToBounds(dronePosX, dronePosY))

        if self.map.isOutOfBounds(dronePosX, dronePosY):
            reward += self.OUT_OF_BOUNDS_PENALTY

        coverage = self.map.getCoverage()
        if coverage > self.END_COVERAGE_THRESH:
            reward += self.COVERAGE_END_REWARD
        return reward

    def step(self, action, permanent = True):
        direction = self._action_to_direction[action]

        dronePosX = self.dronePosX + direction[0]
        dronePosY = self.dronePosY + direction[1]

        # If staying in an already visited place for the past 20 steps, end the game.
        if dronePosX == self.dronePosX and dronePosY == self.dronePosY:
            self.stayStillCnt += 1
        else:
            self.stayStillCnt = 0

        
        outOfBounds = self.map.isOutOfBounds(dronePosX, dronePosY)
        if not outOfBounds:
            self.dronePosX = dronePosX
            self.dronePosY = dronePosY

        observation = self._get_obs()
        info = self._get_info()

        reward = self.getReward(dronePosX, dronePosY, action)


        done = False
        # if self.stayStillCnt > 20:
        #    done = True
        if self.map.getCoverage() > self.END_COVERAGE_THRESH:
            done = True

        if outOfBounds:
            done = True
        else:
            self.map.visitPos(self.dronePosX, self.dronePosY)

        truncated = False

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
