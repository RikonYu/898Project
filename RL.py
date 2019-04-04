import gym
import numpy
from Representations import *
import time
from baselines.a2c import a2c
from baselines.a2c.policies import *
from baselines import deepq
from time import sleep
import IndependentQ
from gui import gameGUI
from models import cnn_rnn_mlp
from baselines.deepq import models
from baselines.ppo1.pposgd_simple import learn
from baselines.trpo_mpi.nosharing_cnn_policy import CnnPolicy
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
import pickle
import matplotlib.pyplot as plt
import baselines.common.tf_util as U

def save(act,env):
    act.save(fname)
    try:
        pickle.dump(env.repr.trail,open('trail'+fname,'wb'))
    except:
        return
def load(env):
    act = deepq.load(path=fname)
    try:
        env.repr.trail=pickle.load(open('trail'+fname,'rb'))
    except:
        pass
    return act

totalsize=8
vision=1
goalSize=1
agentSize=1
#repr=DoubleTrace(vision,totalsize)
#repr=PartialVisionNormal(vision)
repr=PartialAnt(vision,totalsize)
#repr=LearnAnt(vision,totalsize)
fname="deepq-%d-%s-%d-alldir.pkl"%(agentSize,repr.__class__.__name__,totalsize)
env = gameGUI(totalsize, 45, goalSize, agentSize, repr, vision=vision)
qf = models.cnn_to_mlp(convs=[(32, 3, 1), (64, 3, 1), (64, 3, 1)], hiddens=[256])


act,rewards=IndependentQ.learn(env=env,q_func=qf,lr=0.0001,max_timesteps=int(2e5),buffer_size=100000,exploration_fraction=0.2, gamma=0.9, exploration_final_eps=0.2,prioritized_replay=False,print_freq=20)


#act,rewards=IndependentQ.learn(env=env,q_func=qf,lr=0.0001,max_timesteps=int(2e4),buffer_size=100000,exploration_fraction=0.2, gamma=0.9, exploration_final_eps=0.2,prioritized_replay=False,print_freq=20)
pickle.dump(rewards,open('dirtrewards-const.txt','wb'))

save(act,env)
#act = load(env)
# print(env.repr.trail)

for _ in range(10):
    obs, done = env.reset(), False
    episode_rew = 0
    frame=0
    while not done:
        env.render()
        action = []
        qval = []
        for i in range(agentSize):
            prediction = act(np.array(obs[i])[None])
            action.append(prediction[0][0])
            qval.append(prediction[1][0])
            print(qval)
        obs, rew, done, _ = env.step(action, qval)
        episode_rew+=rew
        frame+=1
        time.sleep(0.5)
    #print(env._get_obs())
    time.sleep(10)
    print("Episode reward", episode_rew, "Num Frames:",frame)

'''
env=gameGUI(totalsize,30,1,FullVisionNormal(totalsize),vision=vision)
qf=models.cnn_to_mlp(convs=[(32, 3, 1), (64, 3, 1), (64, 3, 1)],hiddens=[256])
act=IndependentQ.learn(env=env,q_func=qf,lr=0.0001,max_timesteps=int(2e4),buffer_size=10000,exploration_fraction=0.2, gamma=0.9, exploration_final_eps=0.2,prioritized_replay=True,print_freq=20)

'''

def trpo():
    pass

