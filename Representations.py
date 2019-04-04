from gui import *
import numpy
from gym.spaces import *


def visionScope(arr, vis, x, y, default=0):
    if (len(arr.shape) == 2):
        pd = numpy.pad(arr, vis, mode='constant', constant_values=default)
        return pd[x:x + 2 * vis + 1, y:y + 2 * vis + 1].astype(int)
    else:
        ans = numpy.zeros([vis * 2 + 1, vis * 2 + 1, arr.shape[2]])
        for i in range(arr.shape[2]):
            pd = numpy.pad(arr[:, :, i], vis, mode='constant', constant_values=default)
            ans[:, :, i] = pd[x:x + 2 * vis + 1, y:y + 2 * vis + 1].astype(int)
        return numpy.array(ans)


class Representation:
    def __init__(self):
        self.single_channel = None
        self.multiple_channel = None

    def single_repr(self, gui, agent_num=0):
        pass

    def multiple_repr(self, gui, agent_num):
        pass

    def step(self, old_state, action, new_state, value):
        pass


class FullVisionNormal(Representation):
    def __init__(self, totalSize):
        self.single_channel = 3
        self.multiple_channel = 4
        self.action_space = [Discrete(4), Discrete(4)]
        self.observation_space = [
            Box(low=0, high=1, shape=(totalSize, totalSize, 3), dtype=numpy.int32),
            Box(low=0, high=1, shape=(totalSize, totalSize, 4), dtype=numpy.int32)
        ]

    def single_repr(self, gui, agent_num=0):
        return numpy.stack([gui.obstacles, gui.goal_places, gui.hot_agent(0)], axis=2)

    def multiple_repr(self, gui, agent_num):
        return numpy.stack([gui.obstacles, gui.goal_places, gui.all_agent(agent_num), gui.hot_agent(agent_num)], axis=2)


class PartialVisionNormal(Representation):
    def __init__(self, vision):
        self.single_channel = 3
        self.multiple_channel = 4
        self.action_space = [Discrete(4), Discrete(4)]
        self.observation_space = [
            Box(low=0, high=1, shape=(vision * 2 + 1, vision * 2 + 1, 3), dtype=numpy.int32),
            Box(low=0, high=1, shape=(vision * 2 + 1, vision * 2 + 1, 4), dtype=numpy.int32)
        ]

    def single_repr(self, gui, agent_num=0):
        return numpy.stack([visionScope(gui.obstacles, gui.vision, gui.agents[0][0], gui.agents[0][1], 1),
                            visionScope(gui.goal_places, gui.vision, gui.agents[0][0], gui.agents[0][1], 0),
                            gui.hot_agent(0)
                            ], axis=2)

    def multiple_repr(self, gui, agent_num):
        return numpy.stack(
            [visionScope(gui.obstacles, gui.vision, gui.agents[agent_num][0], gui.agents[agent_num][1], 1),
             visionScope(gui.goal_places, gui.vision, gui.agents[agent_num][0], gui.agents[agent_num][1], 0),
             gui.all_agent(agent_num),
             gui.hot_agent(agent_num)
             ], axis=2)


class FullAnt(Representation):
    def __init__(self, totalSize):
        self.single_channel = 4
        self.multiple_channel = 5
        self.action_space = [(Discrete(4)),
                             (Discrete(4))]
        self.observation_space = [
            Box(low=0, high=1, shape=(totalSize, totalSize, 3 + 4), dtype=numpy.int32),
            Box(low=0, high=1, shape=(totalSize, totalSize, 4 + 4), dtype=numpy.int32),
        ]
        self.trail = numpy.zeros([totalSize, totalSize, 4])

    def single_repr(self, gui, agent_num=0):
        return numpy.concatenate([numpy.stack([
            gui.obstacles, gui.goal_places, gui.hot_agent(agent_num)
        ], axis=2), self.trail], axis=2)

    def multiple_repr(self, gui, agent_num):
        return numpy.concatenate([numpy.stack([
            gui.obstacles, gui.goal_places, gui.hot_agent(agent_num), gui.hot_agent(agent_num)
        ], axis=2), self.trail], axis=2)

    def step(self, old_state, action, new_state, value):
        if (type(action) is not list):
            action = [action]
        for i in action:
            old_pos = numpy.transpose(numpy.nonzero(old_state[:, :, 3]))[0]
            self.trail[old_pos, :] += value[i, :]


class PartialAnt(Representation):
    def __init__(self, vision, totalSize):
        self.trailSize = 4
        self.single_channel = 3 + self.trailSize
        self.multiple_channel = 4 + self.trailSize
        self.action_space = [(Discrete(4)),
                             (Discrete(4))]
        self.observation_space = [
            Box(low=0, high=1, shape=(vision * 2 + 1, vision * 2 + 1, self.single_channel), dtype=numpy.int32),
            Box(low=0, high=1, shape=(vision * 2 + 1, vision * 2 + 1, self.multiple_channel), dtype=numpy.int32),
        ]
        self.trail = numpy.zeros([totalSize, totalSize, self.trailSize])

    def single_repr(self, gui, agent_num=0):
        return numpy.concatenate([numpy.stack([
            visionScope(gui.obstacles, gui.vision, gui.agents[0][0], gui.agents[0][1], 1),
            visionScope(gui.goal_places, gui.vision, gui.agents[0][0], gui.agents[0][1], 0),
            gui.hot_agent(0)
        ], axis=2),
            visionScope(self.trail, gui.vision, gui.agents[agent_num][0], gui.agents[agent_num][1], 0)], axis=2)

    def multiple_repr(self, gui, agent_num):
        return numpy.concatenate([
            numpy.stack(
                [visionScope(gui.obstacles, gui.vision, gui.agents[agent_num][0], gui.agents[agent_num][1], 1),
                 visionScope(gui.goal_places, gui.vision, gui.agents[agent_num][0], gui.agents[agent_num][1], 0),
                 gui.all_agent(agent_num),
                 gui.hot_agent(agent_num)], axis=2),
            visionScope(self.trail, gui.vision, gui.agents[agent_num][0], gui.agents[agent_num][1], 0)
        ], axis=2)

    def step(self, old_pos, action, new_pos, value):
        #print(old_pos,new_pos)
        lr = 0.1
        if (type(action) is not list):
            action = [action]
        for i in range(len(action)):
            self.trail[old_pos[i][0], old_pos[i][1], :] *= (1 - lr)
#            self.trail[old_pos[i][0], old_pos[i][1], :] = (1 - lr) * self.trail[old_pos[i][0], old_pos[i][1],
#                                                                     :] + lr * numpy.array(value[i][:])

            if (new_pos[i][0] == old_pos[i][0] - 1):  # up
                self.trail[old_pos[i][0], old_pos[i][1], 0] += lr *-1#value[i][0]

            elif (new_pos[i][1] == old_pos[i][1] - 1):  # left
                self.trail[old_pos[i][0], old_pos[i][1], 1] += lr *-1#value[i][1]

            elif (new_pos[i][1] == old_pos[i][1] + 1):  # right
                self.trail[old_pos[i][0], old_pos[i][1], 2] += lr *-1#value[i][2]

            elif (new_pos[i][0] == old_pos[i][0] + 1):  # down
                self.trail[old_pos[i][0], old_pos[i][1], 3] += lr *-1#value[i][3]



class DoubleTrace(Representation):
    def __init__(self, vision, totalSize):
        self.trailSize=4
        self.trail=numpy.zeros([totalSize,totalSize,self.trailSize])
        self.vision=vision
        self.single_channel=3+self.trailSize
        self.multiple_channel=4+self.trailSize
        self.action_space=[Discrete(4),
                           Discrete(4)]
        if(vision==-1):
            self.observation_space=[Box(low=0,high=1,shape=(totalSize,totalSize,self.single_channel),dtype=numpy.int32),
                                    Box(low=0, high=1, shape=(totalSize,totalSize, self.multiple_channel),
                                        dtype=numpy.int32)
                                    ]
        else:
            self.observation_space=[Box(low=0,high=1,shape=(vision*2+1,vision*2+1,self.single_channel),dtype=numpy.int32),
                                    Box(low=0, high=1, shape=(vision*2+1,vision*2+1, self.multiple_channel),
                                        dtype=numpy.int32)
                                    ]

    def single_repr(self, gui, agent_num=0):
        print('vision```````````````',self.vision)
        if(self.vision==-1):

            return numpy.concatenate([numpy.stack([
                gui.obstacles, gui.goal_places, gui.hot_agent(agent_num)
            ], axis=2), self.trail], axis=2)
        else:
            return numpy.concatenate([numpy.stack([
                visionScope(gui.obstacles, gui.vision, gui.agents[0][0], gui.agents[0][1], 1),
                visionScope(gui.goal_places, gui.vision, gui.agents[0][0], gui.agents[0][1], 0),
                gui.hot_agent(0)
            ], axis=2),
                visionScope(self.trail, gui.vision, gui.agents[agent_num][0], gui.agents[agent_num][1], 0)], axis=2)
    def multiple_repr(self, gui, agent_num):
        if(self.vision==-1):

            return numpy.concatenate([numpy.stack([
                gui.obstacles, gui.goal_places, gui.hot_agent(agent_num)
            ], axis=2), self.trail], axis=2)
        else:
            return numpy.concatenate([numpy.stack([
                visionScope(gui.obstacles, gui.vision, gui.agents[0][0], gui.agents[0][1], 1),
                visionScope(gui.goal_places, gui.vision, gui.agents[0][0], gui.agents[0][1], 0),
                gui.hot_agent(agent_num),
                gui.all_agent(agent_num),
            ], axis=2),
                visionScope(self.trail, gui.vision, gui.agents[agent_num][0], gui.agents[agent_num][1], 0)], axis=2)

    def step(self, old_pos, action, new_pos, vals):
        lr=0.1
        for i in range(len(action)):
            #self.trail[old_pos[i][0], old_pos[i][1], :] = (1 - lr) * self.trail[old_pos[i][0], old_pos[i][1],
            #                                                         :] + lr * numpy.array(value[i][:])

            if (new_pos[i][0] == old_pos[i][0] - 1):  # up
                self.trail[old_pos[i][0], old_pos[i][1], 0] += lr * -1#value[i][0]
                #self.trail[old_pos[i][0], old_pos[i][1], 3] += lr * 1  # value[i][0]

            elif (new_pos[i][1] == old_pos[i][1] - 1):  # left
                self.trail[old_pos[i][0], old_pos[i][1], 1] += lr * -1#value[i][1]
                #self.trail[old_pos[i][0], old_pos[i][1], 2] += lr * 1  # value[i][0]

            elif (new_pos[i][1] == old_pos[i][1] + 1):  # right
                self.trail[old_pos[i][0], old_pos[i][1], 2] += lr * -1#value[i][2]
                #self.trail[old_pos[i][0], old_pos[i][1], 1] += lr * 1  # value[i][0]

            elif (new_pos[i][0] == old_pos[i][0] + 1):  # down
                self.trail[old_pos[i][0], old_pos[i][1], 3] += lr * -1#value[i][3]
                #self.trail[old_pos[i][0], old_pos[i][1], 0] += lr * 1  # value[i][0]
