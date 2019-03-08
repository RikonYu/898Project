import gym
import multiagent
import numpy
from baselines import deepq
from multiagent.policy import InteractivePolicy
from gui import gameGUI
from baselines.deepq import models
def sample(l):
    return [i.sample() for i in l]

def make_env(scenario_name, benchmark=False):
    '''
    Creates a MultiAgentEnv object as env. This can be used similar to a gym
    environment by calling env.reset() and env.step().
    Use env.render() to view the environment on the screen.
    Input:
        scenario_name   :   name of the scenario from ./scenarios/ to be Returns
                            (without the .py extension)
        benchmark       :   whether you want to produce benchmarking data
                            (usually only done during evaluation)
    Some useful env properties (see environment.py):
        .observation_space  :   Returns the observation space for each agent
        .action_space       :   Returns the action space for each agent
        .n                  :   Returns the number of Agents
    '''
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    return env

#env=gym.make('CartPole-v0')
env=gameGUI(64,15,1)
#env=make_env('simple')
#obs_n=env.reset()
qf=models.cnn_to_mlp([(64,3,1),(32,3,1)],[512])
act=deepq.learn(env=env,q_func=qf,lr=0.001,max_timesteps=1000,buffer_size=10000,exploration_fraction=0.1)
act.save("fullFixed.pkl")

for _ in range(10):
    obs, done = env.reset(), False
    episode_rew = 0
    obs, done = env.reset(), False
    episode_rew = 0
    while not done:
        env.render()
        obs, rew, done, _ = env.step(act(obs[None])[0])

    print("Episode reward", episode_rew)
