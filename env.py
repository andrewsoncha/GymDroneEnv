from enum import Enum
import cv2
import numpy as np
import gymnasium as gym
from random import randint
import random
from math import sqrt

'''
Edited by Andrew Chang July 7th, 2026.
Original function was drawRandomCircles (see older branches)

Previously: Draws (circleN) amount of circles of random radius in the range of (0, maxRadius).
            Each cell that is part of a circle is set as value 255.0. 

Now: Draws (circleN) amount of circles of random radius in the range of (0, maxRadius).
     Each cell that is part of a circle is set as a floating point number set between 0 and 255.
     The cell values are higher the close they are to the center of its circle. 
     The equation for the cell value is 255 * ( 1 - distance_from_center / maxRadius)
'''
def drawProbabilityCircles(imageShape, circleN, maxRadius):
    image = np.zeros(imageShape, dtype=np.uint8)
    width, height = imageShape
    for i in range(circleN):
        center = (randint(0, width), randint(0, height))
        radius = randint(0, maxRadius)

        cv2.circle(image, center, radius, 255, -1)
    dist = cv2.distanceTransform(image, cv2.DIST_L2, 5)
    min_dist, max_dist = np.min(dist), np.max(dist)
    norm_dist = (dist - min_dist)/(max_dist - min_dist)*255
    return norm_dist 

def drawBinaryCircles(imageShape, circleN, maxRadius):
    image = np.zeros(imageShape, dtype=np.uint8)
    width, height = imageShape
    for i in range(circleN):
        center = (randint(0, width), randint(0, height))
        radius = randint(0, maxRadius)

        cv2.circle(image, center, radius, 255, -1)
    return image 


class Map:
    def __init__(self, mapLen, circleN, visionRange = 5, seed=None):
        if seed is not None:
            random.seed(seed)

        # self.img = drawProbabilityCircles((mapLen, mapLen), circleN, mapLen//4)
        self.img = drawBinaryCircles((mapLen, mapLen), circleN, mapLen//4)
        self.img_sum = np.count_nonzero(self.img)
        maxVal = np.max(self.img)
        self.visit = np.zeros_like(self.img)
        self.imgBiggerSize = np.zeros_like
        self.rowN = self.img.shape[0]
        self.colN = self.img.shape[1]
        self.visionRange = visionRange
        self.paddedRowN = self.rowN+self.visionRange*2
        self.paddedColN = self.colN+self.visionRange*2
        self.paddedImg = np.zeros(shape = (self.paddedRowN, self.paddedColN), dtype=np.uint8)
        self.paddedVisit = np.zeros_like(self.paddedImg) 
        self.paddedVisitMask = np.zeros_like(self.paddedImg)
        self.paddedVisitImg = np.zeros_like(self.paddedImg)
        self.paddedImg[self.visionRange: self.rowN+self.visionRange, self.visionRange: self.colN +self.visionRange] = self.img

    def isOutOfBounds(self, posX, posY):
        if posX < 0 or posX >= self.colN :
            return True
        if posY < 0 or posY >= self.rowN:
            return True
        return False

    # June 23rd, 2026. - Andrew Chang
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

    '''
    def visitPos(self, posX, posY):
        for x in range(posX-self.visionRange, posX+self.visionRange+1):
            for y in range(posY-self.visionRange, posY+self.visionRange+1):
                paddedX = posX + self.visionRange
                paddedY = posY + self.visionRange
                if self.paddedVisit[paddedX][paddedY] < 255:
                    self.paddedVisit[paddedX][paddedY] += 1
                self.paddedVisitMask[paddedX-self.visionRange:paddedX+self.visionRange, paddedY-self.visionRange:paddedY+self.visionRange] = 255
                self.paddedVisitImg[paddedX-self.visionRange:paddedX+self.visionRange, paddedY-self.visionRange:paddedY+self.visionRange] = self.paddedImg[paddedX-self.visionRange:paddedX+self.visionRange, paddedY-self.visionRange:paddedY+self.visionRange]
    '''
    def visitPos(self, posX, posY):
        paddedX = posX + self.visionRange
        paddedY = posY + self.visionRange
        for x in range(paddedX - self.visionRange, paddedX + self.visionRange + 1):
            for y in range(paddedY - self.visionRange, paddedY + self.visionRange + 1):
                if self.paddedVisit[x][y] < 255:
                    self.paddedVisit[x][y] += 1
        self.paddedVisitMask[
            paddedX-self.visionRange:paddedX+self.visionRange+1,
            paddedY-self.visionRange:paddedY+self.visionRange+1
        ] = 255
        self.paddedVisitImg[
            paddedX-self.visionRange:paddedX+self.visionRange+1,
            paddedY-self.visionRange:paddedY+self.visionRange+1
        ] = self.paddedImg[
            paddedX-self.visionRange:paddedX+self.visionRange+1,
            paddedY-self.visionRange:paddedY+self.visionRange+1
        ]

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

    def getCenteredGlobalView(self, droneX, droneY):
        startX, startY = self.rowN-droneX, self.colN-droneY
        endX, endY = startX+self.rowN, startY+self.colN
        globalView = np.zeros((self.rowN*2, self.colN*2), np.uint8)
        globalView[startX:endX, startY:endY] = self.img
        return globalView
    
    def getCenteredVisit(self, droneX, droneY):
        startX, startY = self.rowN-droneX, self.colN-droneY
        endX, endY = startX+self.rowN, startY+self.colN
        globalVisit  = np.zeros((self.rowN*2, self.colN*2), np.uint8)
        # globalView[startX:endX, startY:endY] = self.img
        globalVisit[startX:endX, startY:endY] = self.paddedVisitMask[self.visionRange:self.rowN+self.visionRange, self.visionRange:self.colN+self.visionRange]
        return globalVisit

    '''
    Modified on July 8th, 2026 by Andrew Chang.

    Originally, the environment only allowed discrete values in maps.
    If a cell was a target zone, it was a 1, if else, it was a 0.
    Because of this, coverage was #(covered target zone cells) / #(total target zone cells)
    Now, the cells can have any real value in between 0.0 and 1.0. 
    Because of this, coverage is now defined as sum(covered target zone cells) / sum(all cells)
    '''
    def getCoverage(self):
        covered_sum = np.sum(self.paddedVisitImg) / 255
        coverage = covered_sum / self.img_sum
        return coverage

class Actions(Enum):
    UP = 0
    DOWN = 1
    RIGHT = 2
    LEFT = 3
    HOVER = 4

class Env(gym.Env):
    VISION_RANGE = 7 

    MAP_LEN = 50
    MAP_CIRCLE_N = 5

    # Reward constants matched to uavSim "Learning to Recharge" structure (without charging).
    # Edited by Andrew Chang [DATE].
    # uavSim equivalent names shown in comments.
    MOVEMENT_PENALTY   = -0.2    # movement_penalty:       small per-step cost to encourage efficiency
    OOB_PENALTY        = -150.0  # empty_battery_penalty:  large penalty for going out of bounds / crashing
    CELL_REWARD        =  0.4    # cell_reward:            reward per unit of newly covered target area
    STAY_STILL_PENALTY = -0.4

    EXPLORATION_REWARD = 0.05  # per newly visited cell regardless of type
    CELL_REWARD        = 0.4   # per newly visited target cell (on top of exploration reward)

    # Episode termination
    END_COVERAGE_THRESH  = 0.85
    TOTAL_TIMESTEP_CAP   = 8 * MAP_LEN * MAP_LEN

    def _get_obs(self):
        local_map, local_visit = self.map.getLocalView(self.dronePosX, self.dronePosY)
        global_map = self.map.getCenteredGlobalView(self.dronePosX, self.dronePosY)
        global_visit = self.map.getCenteredVisit(self.dronePosX, self.dronePosY)
        observation = {
                'drone_pos': np.array([self.dronePosX/self.map.colN, self.dronePosY/self.map.rowN]),
                'local_map': local_map/255.0,
                'global_map': global_map/255.0,
                'local_visit': local_visit/255.0,
                'global_visit': global_visit/255.0,
                }
        return observation

    def _get_info(self):
        local_map, local_visit = self.map.getLocalView(self.dronePosX, self.dronePosY)
        global_map = self.map.getCenteredGlobalView(self.dronePosX, self.dronePosY)
        global_visit = self.map.getCenteredVisit(self.dronePosX, self.dronePosY)
        imgVal = self.map.getImgValue(self.dronePosX, self.dronePosY)
        return {
                "drone_pos": (self.dronePosX, self.dronePosY),
                "local_map": local_map,
                "global_map": global_map,
                "local_visit": local_visit,
                'global_visit': global_visit,
                "current_val": imgVal
                }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.map = Map(mapLen = self.MAP_LEN, circleN = self.MAP_CIRCLE_N, visionRange = self.VISION_RANGE, seed=self.map_seed)
        self.dronePosX = self.map.colN//2
        self.dronePosY = self.map.rowN//2
        self.stayStillCnt = 0
        self.envTimestep = 0
        self.prevCoverage = 0

        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def __init__(self, render_mode="", map_seed=None):
        self.map_seed = map_seed
        self.render_mode = render_mode
        self.map = Map(mapLen = self.MAP_LEN, circleN = self.MAP_CIRCLE_N, visionRange = self.VISION_RANGE, seed=map_seed)
        self.dronePosX = self.map.colN//2
        self.dronePosY = self.map.rowN//2
        self.action_space = gym.spaces.Discrete(5)
        self.envTimestep = 0
        self.prevCoverage = 0
        self.observation_space = gym.spaces.Dict(
                {
                    "local_map": gym.spaces.Box(low=0, high=1.0, shape=(self.VISION_RANGE*2+1, self.VISION_RANGE*2+1), dtype=np.float64),
                    "local_visit": gym.spaces.Box(low=0, high=1.0, shape=(self.VISION_RANGE*2+1, self.VISION_RANGE*2+1), dtype=np.float64),
                    "global_map": gym.spaces.Box(low=0, high=1.0, shape=(self.MAP_LEN*2, self.MAP_LEN*2), dtype=np.float64),
                    "global_visit": gym.spaces.Box(low=0, high=1.0, shape=(self.MAP_LEN*2, self.MAP_LEN*2), dtype=np.float64),
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

    '''
    Reward function matched to uavSim "Learning to Recharge" structure (without charging).
    Edited by Andrew Chang [DATE].

    Three components only:
      1. MOVEMENT_PENALTY  — small negative reward every step (encourages efficiency)
      2. OOB_PENALTY       — large negative reward when drone attempts to leave the map
      3. CELL_REWARD       — positive reward proportional to newly covered target area,
                             applied in step() after visitPos() updates the coverage map.
    '''
    def getReward(self, dronePosX, dronePosY):
        reward = self.MOVEMENT_PENALTY

        if self.map.isOutOfBounds(dronePosX, dronePosY):
            reward += self.OOB_PENALTY

        if self.stayStillCnt > 0:
            reward += self.STAY_STILL_PENALTY * self.stayStillCnt

        return reward

    def step(self, action, permanent=True):
        direction = self._action_to_direction[action]

        dronePosX = self.dronePosX + direction[0]
        dronePosY = self.dronePosY + direction[1]

        if dronePosX == self.dronePosX and dronePosY == self.dronePosY:
            self.stayStillCnt += 1
        else:
            self.stayStillCnt = 0

        outOfBounds = self.map.isOutOfBounds(dronePosX, dronePosY)
        if not outOfBounds:
            self.dronePosX = dronePosX
            self.dronePosY = dronePosY

        # Base reward: movement penalty + OOB penalty if applicable.
        # Uses attempted position so OOB penalty fires correctly.
        reward = self.getReward(dronePosX, dronePosY)

        done = False
        if self.envTimestep > self.TOTAL_TIMESTEP_CAP:
            done = True
        if self.map.getCoverage() > self.END_COVERAGE_THRESH:
            done = True
        else:
            # Cell reward: proportional to newly covered target area this step.
            # Computed after visitPos so coverage delta reflects this step only.
            # Equivalent to uavSim's cells_collected * cell_reward, but continuous
            # because our map uses probability values rather than binary cells.
            '''
            prev_coverage = self.map.getCoverage()
            self.map.visitPos(self.dronePosX, self.dronePosY)
            new_coverage = self.map.getCoverage()
            newly_covered = (new_coverage - prev_coverage) * self.map.img_sum
            reward += self.CELL_REWARD * newly_covered
            '''
            prev_coverage = self.map.getCoverage()
            prev_visited = np.sum(self.map.paddedVisitMask)
            self.map.visitPos(self.dronePosX, self.dronePosY)
            new_coverage = self.map.getCoverage()
            new_visited = np.sum(self.map.paddedVisitMask)

            newly_visited = (new_visited - prev_visited) / 255
            newly_covered_targets = (new_coverage - prev_coverage) * self.map.img_sum

            reward += self.EXPLORATION_REWARD * newly_visited
            reward += self.CELL_REWARD * newly_covered_targets

        self.envTimestep += 1

        observation = self._get_obs()
        info = self._get_info()

        truncated = False

        return observation, float(reward), done, truncated, info

    def render(self, render_mode='rgb_array'):
        colorImage = cv2.merge([self.map.paddedImg, self.map.paddedImg, self.map.paddedImg])
        zeros = np.zeros_like(self.map.paddedVisit)
        redPath = cv2.merge([np.ones_like(self.map.paddedVisit)*255, zeros, self.map.paddedVisit])
        coverageMap = cv2.merge([self.map.paddedVisitMask, self.map.paddedVisitMask, self.map.paddedVisitMask])
        _, mask = cv2.threshold(self.map.paddedVisit, 1, 255, cv2.THRESH_BINARY)
        mask = mask/255
        maskColor = cv2.merge([mask, mask, mask])
        frame = colorImage
        for c in range(3):
            frame[:,:,c] = (colorImage[:,:,c]*(1-mask) + redPath[:,:,c]*mask)*0.8 + coverageMap[:,:,c]*0.2
        resizedFrame = cv2.resize(frame, (500, 500))
        resizedCoverageMap = cv2.resize(coverageMap, (500, 500))
        cv2.imshow('resizedCoverageMap', resizedCoverageMap)
        cv2.waitKey(1)
        return resizedFrame


if __name__ == '__main__':
    env = Env()
