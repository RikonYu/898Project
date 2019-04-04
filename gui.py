import numpy
import time
import gym
from gym.spaces import Box
import copy
from gym.envs.registration import EnvSpec
from gym import spaces
from gym.envs.classic_control import rendering
import numpy as np
from gym.utils import seeding
import pickle
import os.path


def isBlack(x):
    return x.vec4[0]+x.vec4[1]<=0.1
def generateMaze(size):  # 1 = obstacle, 0 = blank
    path = 'maze%d.txt'%size
    if (os.path.isfile(path)):
        return pickle.load(open(path, 'rb'))
    ans = numpy.zeros([size, size])

    def _gen(top, bot, left, right):
        if (left + 1 >= right or top + 1 >= bot):
            return
        mid = [numpy.random.choice(numpy.arange(top + 1, bot)), numpy.random.choice(numpy.arange(left + 1, right))]
        if (left + 1 < right):
            ans[top:bot, mid[1]] = 1
            ans[numpy.random.choice(numpy.arange(top, mid[0])), mid[1]] = 0
            ans[numpy.random.choice(numpy.arange(mid[0], bot)), mid[1]] = 0
        if (top + 1 < bot):
            ans[mid[0], left:right] = 1
            ans[mid[0], numpy.random.choice(numpy.arange(left, mid[1]))] = 0
            ans[mid[0], numpy.random.choice(numpy.arange(mid[1], right))] = 0

        _gen(top, mid[0] - 1, left, mid[1] - 1)
        _gen(mid[0] + 1, bot, left, mid[1] - 1)
        _gen(top, mid[0] - 1, mid[1] + 1, right)
        _gen(mid[0] + 1, bot, mid[1] + 1, right)

    _gen(0, size, 0, size)
    pickle.dump(ans, open(path, 'wb'))
    return ans

def generateAG(map, goalSize,agentSize, goalPos=None, agentPos=None):
    if(goalPos!=None):
        return numpy.array([goalPos]), numpy.array([agentPos])
    goods = numpy.argwhere(map == 0)
    goals = goods[numpy.random.choice(len(goods), size=goalSize+agentSize, replace=False)]
    return goals[:goalSize], goals[goalSize:]


class gameGUI(gym.Env):
    def __init__(self, totalSize, cellSize, goalSize,agentSize,repr, vision=-1, title=''):
        self.timeFrame = 0
        self.repr=repr
        self.num_envs=1
        self.maxFrame = totalSize * totalSize * goalSize * 2
        self.totalSize = totalSize
        self.vision = vision
        self.cellSize = cellSize
        self.goalSize = goalSize
        self.agentSize= agentSize
        self.is_single = 0
        self.viewer = None
        self.trace=numpy.zeros([totalSize,totalSize])
        self.obstacles = numpy.zeros([totalSize, totalSize]).astype(numpy.int32)
        self.agents = numpy.zeros([self.agentSize, 2]).astype(numpy.int32)
        self.goals = numpy.zeros([self.goalSize, 2]).astype(numpy.int32)
        self.goal_places = numpy.zeros([totalSize, totalSize]).astype(numpy.int32)
        self.goal_reached=numpy.zeros([self.goalSize])
        self.agent_places = numpy.zeros([totalSize, totalSize]).astype(numpy.int32)
        if (self.is_single):
            self.action_space = self.repr.action_space[0]
            self.observation_space=self.repr.observation_space[0]
        else:
            self.action_space = [self.repr.action_space[1] for i in range(self.agentSize)]
            self.observation_space = [self.repr.observation_space[1] for i in range(self.agentSize)]
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _updateBG(self):
        if (self.vision == -1):
            return
        for i in range(self.totalSize):
            for j in range(self.totalSize):
                inSight = 0
                for k in range(self.agentSize):
                    if (abs(i - self.agents[k][0]) <= self.vision and abs(j - self.agents[k][1]) <= self.vision):
                        inSight = 1
                if (isBlack(self.backGeom[i][j]._color)==False):
                    if (inSight == 1):
                        self.backGeom[i][j].set_color(1.0, 1.0, 1.0)
                    else:
                        self.backGeom[i][j].set_color(0.5,0.5,0.5)

    def _fillSquare(self, left, top, color):
        length = self.cellSize
        left *= self.cellSize
        top *= self.cellSize
        x = rendering.FilledPolygon(
            [(left, top), (left + length, top), (left + length, top + length), (left, top + length)])
        x.set_color(color[0] / 255.0, color[1] / 255.0, color[2] / 255.0)
        self.viewer.add_geom(x)
        return x

    def _updateGeom(self, geom, ob):
        try:
            top, left = ob[0], ob[1]
            top *= self.cellSize
            left *= self.cellSize
            length = self.cellSize
            geom.v = ([(left, top), (left + length, top), (left + length, top + length), (left, top + length)])
        except:
            pass
    def render(self):
        if (self.viewer == None):
            self.viewer = rendering.Viewer(self.totalSize * self.cellSize * 3, self.totalSize * self.cellSize * 2)
            self.agentGeom = [None for i in range(self.agentSize)]
            self.goalGeom = [None for i in range(self.goalSize)]
            self.trailGeom = [[[None for i in range(4)] for j in range(self.totalSize)] for k in range(self.totalSize)]
            self.backGeom = [[None for i in range(self.totalSize)] for j in range(self.totalSize)]
            for i in range(self.totalSize):
                for j in range(self.totalSize):
                    if (self.obstacles[i][j] == 1):
                        self.backGeom[i][j] = self._fillSquare(j, i, (0, 0, 0))
                    else:
                        self.backGeom[i][j] = self._fillSquare(j, i, (255,255,255))
            for i in range(self.goalSize):
                self.goalGeom[i] = self._fillSquare(self.goals[i][1], self.goals[i][0], (0, 255, 0))

            for i in range(self.agentSize):
                self.agentGeom[i] = self._fillSquare(self.agents[i][1], self.agents[i][0], (255.0, 0, 0))

            for i in range(self.totalSize):
                for j in range(self.totalSize):
                    for k in range(4):
                        self.trailGeom[i][j][k] = self._fillSquare(j+self.totalSize*((k+1)%3), i+self.totalSize*((k+1)//3), (255, 255, 255))
            for i in range(1,3):
                l=rendering.Line((i*self.totalSize*self.cellSize,0),
                                                    (i*self.totalSize*self.cellSize,2*self.totalSize*self.cellSize))
                l.set_color(1,0,0)
                self.viewer.add_geom(l)
            l=rendering.Line((0,self.totalSize*self.cellSize),
                             (3*self.totalSize*self.cellSize,self.totalSize*self.cellSize))
            l.set_color(1, 0, 0)
            self.viewer.add_geom(l)


        if(hasattr(self.repr,'trail')):
            maxt=numpy.amax(self.repr.trail)+1e-6
            mint=numpy.amin(self.repr.trail)
            for i in range(self.totalSize):
                for j in range(self.totalSize):
                     for k in range(self.repr.trailSize):
                        self.trailGeom[i][j][k].set_color((self.repr.trail[i, j, k] - mint) / (maxt - mint),
                                                       (self.repr.trail[i, j, k] - mint) / (maxt - mint),
                                                       (self.repr.trail[i, j, k] - mint) / (maxt - mint))


        for i in range(len(self.agents)):
            self._updateGeom(self.agentGeom[i], self.agents[i])
        for i in range(len(self.goals)):
            self._updateGeom(self.goalGeom[i], self.goals[i])
        self._updateBG()
        return self.viewer.render(return_rgb_array=0)

    def markAgents(self):
        self.agent_places = numpy.zeros([self.totalSize, self.totalSize]).astype(numpy.int32)
        for i in range(len(self.agents)):
            self.agent_places[self.agents[i, 0], self.agents[i, 1]] = 1

    def hot_agent(self, ind):
        if(self.vision==-1):
            ans = numpy.zeros([self.totalSize, self.totalSize]).astype(numpy.int32)
            ans[self.agents[ind][0], self.agents[ind][1]] = 1
            return ans
        else:
            ans=numpy.zeros([self.vision*2+1,self.vision*2+1]).astype(numpy.int32)
            ans[self.vision,self.vision]=1
            return ans

    def all_agent(self,k):
        #print(self.agents,k,self.agentSize)
        if(self.vision==-1):
            ans = numpy.zeros([self.totalSize, self.totalSize]).astype(numpy.int32)
            for ind in range(self.agentSize):
                if(k!=ind):
                    ans[self.agents[ind][0], self.agents[ind][1]] = 1
            return ans
        else:
            ans=numpy.zeros([self.vision*2+1,self.vision*2+1]).astype(numpy.int32)
            for ind in range(self.agentSize):
                if(k!=ind):
                    if (abs(self.agents[ind][0] - self.agents[k][0]) <= self.vision and
                            abs(self.agents[ind][1] - self.agents[k][1]) <= self.vision):
                        ans[self.agents[ind][0] - self.agents[k][0]+self.vision, self.agents[ind][1] - self.agents[k][1]+self.vision]=1
            return ans

    def _get_obs(self):
        if(self.is_single):
            return self.repr.single_repr(self)
        else:
            return [self.repr.multiple_repr(self,i) for i in range(self.agentSize)]

    def reset(self, goalPos=None, agentPos=None):
        self.timeFrame = 0
        self.agents = numpy.zeros([self.agentSize, 2]).astype(numpy.int32)
        self.goals = numpy.zeros([self.goalSize, 2]).astype(numpy.int32)
        self.obstacles = numpy.zeros([self.totalSize, self.totalSize]).astype(numpy.int32)
        self.goal_places = numpy.zeros([self.totalSize, self.totalSize]).astype(numpy.int32)
        self.agent_places = numpy.zeros([self.totalSize, self.totalSize]).astype(numpy.int32)
        self.goal_reached=numpy.zeros([self.goalSize]).astype(numpy.int32)
        self.obstacles = generateMaze(self.totalSize)
        self.goals, self.agents = generateAG(self.obstacles, self.goalSize,self.agentSize,goalPos=goalPos,agentPos=agentPos)
        self.markAgents()

        self.last_reward=0
        if (self.viewer):
            self.viewer.close()
        self.viewer = None
        self.tviewer=None
        for i in self.goals:
            self.goal_places[i[0]][i[1]] = 1
        #self.repr.trail*=0.9
        return self._get_obs()

    def step(self, actions,vals=None):
        if (self.is_single):
            actions = [actions]
        old_pos=copy.deepcopy(self.agents)
        newval=[]
        for i in range(len(actions)):
            if(type(actions[i])==numpy.int64): #action in {0,1,2,3}
                if (actions[i] == 0 and self.agents[i][0] != 0):  # up
                    if (self.obstacles[self.agents[i][0] - 1][self.agents[i][1]] != 1):
                        self.agents[i][0] -= 1
                elif (actions[i] == 1 and self.agents[i][1] != 0):  # left
                    if (self.obstacles[self.agents[i][0]][self.agents[i][1] - 1] != 1):
                        self.agents[i][1] -= 1
                elif (actions[i] == 2 and self.agents[i][1] != self.totalSize - 1):  # right
                    if (self.obstacles[self.agents[i][0]][self.agents[i][1] + 1] != 1):
                        self.agents[i][1] += 1
                elif (actions[i] == 3 and self.agents[i][0] != self.totalSize - 1):  # down
                    if (self.obstacles[self.agents[i][0] + 1][self.agents[i][1]] != 1):
                        self.agents[i][0] += 1
            elif(isinstance(actions[i],(tuple,list))==True):#action=(move,val)
                if (actions[i][0] == 0 and self.agents[i][0] != 0):  # up
                    if (self.obstacles[self.agents[i][0] - 1][self.agents[i][1]] != 1):
                        self.agents[i][0] -= 1
                elif (actions[i][0] == 1 and self.agents[i][1] != 0):  # left
                    if (self.obstacles[self.agents[i][0]][self.agents[i][1] - 1] != 1):
                        self.agents[i][1] -= 1
                elif (actions[i][0] == 2 and self.agents[i][1] != self.totalSize - 1):  # right
                    if (self.obstacles[self.agents[i][0]][self.agents[i][1] + 1] != 1):
                        self.agents[i][1] += 1
                elif (actions[i][0] == 3 and self.agents[i][0] != self.totalSize - 1):  # down
                    if (self.obstacles[self.agents[i][0] + 1][self.agents[i][1]] != 1):
                        self.agents[i][0] += 1
                newval.append([actions[i][1:]])
        self.markAgents()
        if(vals!=None):
            self.repr.step(old_pos,actions,self.agents[:],vals)
        else:
            self.repr.step(old_pos,actions,self.agents,newval)
        reward = -float(self.goalSize)/self.maxFrame

        self.timeFrame += 1
        for i in range(len(self.goals)):
            if (self.agent_places[self.goals[i][0], self.goals[i][1]] == 1):
                reward += self.goal_reached[i]==0
                self.goal_reached[i]=1
        done = (numpy.sum(self.goal_reached)== self.goalSize or self.timeFrame >= self.maxFrame)
        #self.last_reward=reward
        return self._get_obs(), reward, done, {}
