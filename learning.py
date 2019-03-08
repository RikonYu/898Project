import numpy
import keras
import pickle
import tensorflow as tf
from baselines.deepq.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer

class MultiAct:
    def __init__(self,acts):
        self.acts=acts
    def step(self,observations,**kwargs):
        act=[]
        for obs in range(len(observations)):
            act.append(self.act[obs](observations[obs],**kwargs))
        return act,None,None,None
    def save(self,fname='MultiAct'):
        pickle.dump(open(fname,'wb'),self.acts)
    @staticmethod
    def load(fname='MultiAct'):
        x=pickle.load(open(fname,'rb'))
        return MultiAct(x)

def learn(env):
    pass