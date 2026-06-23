from enum import Enum
import cv2
import numpy as np
import gymnasium as gym
from random import randint
import random

def drawRandomCircles(imageShape, circleN, maxRadius):
    image = np.zeros(imageShape, dtype=np.uint8)
    width, height = imageShape
    for i in range(circleN):
        center = (randint(0, width), randint(0, height))
        radius = randint(0, maxRadius)
        cv2.circle(image, center, radius, color=255, thickness=-1)
    return image

class Map:
    def __init__(self, visionRange = 5, seed=None):
        if seed is not None:
            random.seed(seed)

        self.img = drawRandomCircles((300, 300), 60, 35)
        maxVal = np.max(self.img)
        self.visit = np.zeros_like(self.img)
        self.imgBiggerSize = np.zeros_like
        self.rowN = self.img.shape[0]
        self.colN = self.img.shape[1]
        print('self.rowN: ', self.rowN)
        print('self.colN: ', self.colN)
        self.visionRange = visionRange
        self.paddedRowN = self.rowN+self.visionRange*2
        self.paddedColN = self.colN+self.visionRange*2
        self.paddedImg = np.zeros(shape = (self.paddedRowN, self.paddedColN), dtype=np.uint8)
        self.paddedVisit = np.zeros_like(self.paddedImg) 
        self.paddedImg[self.visionRange: self.rowN+self.visionRange, self.visionRange: self.colN +self.visionRange] = self.img

    def isOutOfBounds(self, posX, posY):
        if posX < 0 or posX >= self.colN :
            return True
        if posY < 0 or posY > self.rowN:
            return True
        return False

    # June 23st, 2026. - Andrew Chang
    # This function probably shouldn't be used anymore
    # This was written when the environment actively punished when the model got near the edge instead of just having it be impossible to go out of bounds
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
            paddedX = posX + self.visionRange
            paddedY = posY + self.visionRange
            return self.paddedImg[paddedX, paddedY]

    def visitPos(self, posX, posY):
        for x in range(posX-self.visionRange, posX+self.visionRange+1):
            for y in range(posY-self.visionRange, posY+self.visionRange+1):
                paddedX = posX + self.visionRange
                paddedY = posY + self.visionRange
                if self.paddedVisit[x][y] < 255:
                    self.paddedVisit[x][y] += 1

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
        paddedX = posX + self.visionRange
        paddedY = posY + self.visionRange
        mapView = self.paddedImg[paddedX-self.visionRange : paddedX+self.visionRange+1, paddedY-self.visionRange : paddedY+self.visionRange+1]
        visitView = self.paddedVisit[paddedX-self.visionRange : paddedX+self.visionRange+1, paddedY-self.visionRange : paddedY+self.visionRange+1]
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

        # June 2nd, 2026 - Andrew Chang.
        # Originally, the below part was 
        # old_cells_cnt = self.visionRange**2 - new_cells_seen_cnt
        # However, we now subtract new_cells_seen_cnt from self.visionRange instead of self.visionRange**2
        # because we want to know among the "newly seen edge" cells that are newly seen while moving
        # how many have been previously seen and how many have not.
        # Because of this, the maximum amount of "newly seen edge" cells are the length of
        # a column or a row, which is self.visionRange.
        # We add the first if statement in case of the initial step, where every cell
        # in range (self.visionRange**2) is a newly seen cell.
        if len(newlySeenCoor) == self.visionRange**2:
            old_cells_cnt = 0
        else:
            old_cells_cnt = self.visionRange - new_cells_seen_cnt

        return new_cells_seen_cnt, new_target_seen, new_nonTarget_seen, old_cells_cnt

    def getCoverage(self):
        visit_size = len(self.visit) * len(self.visit[0])
        visit_uniques, visit_counts = np.unique(self.visit, return_counts=True)
        visit_countDict = dict(zip(visit_uniques, visit_counts))
        
        unseen_cnt = 0
        seen_cnt = 0

        if 0 in visit_countDict:
            unseen_cnt = visit_countDict[0]
        seen_cnt = visit_size - unseen_cnt

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
    NEW_NONTARGET_REWARD = 0.015
    NEW_TARGET_REWARD = 1.0
    ALREADY_SEEN_PENALTY = -0.005
    CLOSE_TO_BOUNDS_PENALTY = -5.0
    END_COVERAGE_THRESH = 0.85
    COVERAGE_END_REWARD = 5.0
    OUT_OF_BOUNDS_PENALTY = -10.0
    VISIT_PENALTY = -0.02
    HOVER_PENALTY = -0.1
    STAY_STILL_PENALTY = -0.1
    COVERAGE_DELTA_REWARD = 10.0
    TOTAL_TIMESTEP_CAP = 950

    def _get_obs(self):
        local_map, local_visit = self.map.getLocalView(self.dronePosX, self.dronePosY)
        observation = {
                'drone_pos': np.array([self.dronePosX/self.map.colN, self.dronePosY/self.map.rowN]),
                'local_map': local_map/255.0,
                'local_visit': local_visit/255.0
                }
        return observation

    def _get_info(self):
        local_map, local_visit = self.map.getLocalView(self.dronePosX, self.dronePosY)
        imgVal = self.map.getImgValue(self.dronePosX, self.dronePosY)
        return {
                "drone_pos": (self.dronePosX, self.dronePosY),
                "local_map": local_map,
                "local_visit": local_visit,
                "current_val": imgVal
                }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.map = Map(visionRange = self.VISION_RANGE, seed=self.map_seed)
        self.dronePosX = self.map.colN//2
        self.dronePosY = self.map.rowN//2
        self.stayStillCnt = 0
        self.envTimestep = 0
        # self.map.visit = np.zeros_like(self.map.img)
        #self.map.visitPos(self.dronePosX, self.dronePosY)

        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def __init__(self, render_mode="", map_seed = None):
        self.map_seed = map_seed
        self.render_mode = render_mode
        self.map = Map(visionRange = self.VISION_RANGE, seed=map_seed)
        self.dronePosX = self.map.colN//2
        self.dronePosY = self.map.rowN//2
        self.action_space = gym.spaces.Discrete(5)
        self.envTimestep = 0
        self.observation_space = gym.spaces.Dict(
                {
                    "local_map": gym.spaces.Box(low=0, high=1.0, shape=(self.VISION_RANGE*2+1, self.VISION_RANGE*2+1), dtype=np.float64),
                    "local_visit": gym.spaces.Box(low=0, high=1.0, shape=(self.VISION_RANGE*2+1, self.VISION_RANGE*2+1), dtype=np.float64),
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

        if self.stayStillCnt > 2:
            reward += self.STAY_STILL_PENALTY * (self.stayStillCnt - 2)

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

        reward = self.getReward(dronePosX, dronePosY, action)

        done = False
        if self.envTimestep > self.TOTAL_TIMESTEP_CAP:
            done = True
        if self.map.getCoverage() > self.END_COVERAGE_THRESH:
            done = True

        if outOfBounds:
            #done = True
            reward += OUT_OF_BOUNDS_PENALTY
            pass
        else:
            # prev_coverage and new_coverage done from claude suggestion to incentivize more exploration by the RL agent
            prev_coverage = self.map.getCoverage()
            self.map.visitPos(self.dronePosX, self.dronePosY)
            new_coverage = self.map.getCoverage()
            reward += self.COVERAGE_DELTA_REWARD * (new_coverage - prev_coverage)

        observation = self._get_obs()
        info = self._get_info()

        truncated = False

        return observation, float(reward), done, truncated, info

    def render(self, render_mode='rgb_array'):
        colorImage = cv2.merge([self.map.paddedImg, self.map.paddedImg, self.map.paddedImg])
        zeros = np.zeros_like(self.map.paddedVisit)
        redPath = cv2.merge([np.ones_like(self.map.paddedVisit)*255, zeros, self.map.paddedVisit])
        _, mask = cv2.threshold(self.map.paddedVisit, 1, 255, cv2.THRESH_BINARY)
        mask = mask/255
        maskColor = cv2.merge([mask, mask, mask])
        frame = colorImage
        for c in range(3):
            frame[:,:,c] = colorImage[:,:,c]*(1-mask) + redPath[:,:,c]*mask
        resizedFrame = cv2.resize(frame, (500, 500))
        return resizedFrame


if __name__ == '__main__':
    env = Env()
    cv2.waitKey(1000)
