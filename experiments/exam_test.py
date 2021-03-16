import os
import gym
import numpy as np
gym.logger.set_level(40)
import tensorflow as tf

os.environ['GYM_CONFIG_CLASS'] = 'Example'
os.environ['GYM_CONFIG_PATH'] = '../maca/configs/my_config.py'

from maca.agents.distributedagent import DistributedAgent as Agent
from maca.agents.obstacle1 import Obstacle as obstacle1
# Policies
from maca.policies import FormationPolicy
from maca.policies import StaticPolicy
from maca.policies import NonCooperativePolicy
from maca.policies import RVOPolicy
from maca.policies import CADRLPolicy
from maca.policies import GA3CCADRLPolicy
from maca.policies import ExternalPolicy
from maca.policies import LearningPolicy
from maca.policies import CARRLPolicy
from maca.policies import LearningPolicyGA3C
# Dynamics
from maca.envs.dynamics.FullDynamics import FullDynamics
# Sensors
from maca.envs.sensors.OtherAgentsStatesSensor import OtherAgentsStatesSensor

from maca.configs.config_test import MyDict

policy_dict = {
    'RVO': RVOPolicy,
    'noncoop': NonCooperativePolicy,
    'carrl': CARRLPolicy,
    'external': ExternalPolicy,
    'GA3C_CADRL': GA3CCADRLPolicy,
    'learning': LearningPolicy,
    'learning_ga3c': LearningPolicyGA3C,
    'static': StaticPolicy,
    'CADRL': CADRLPolicy,
    'Formation': FormationPolicy,
}
agent_num = 4
start_offset = [0.1, 0.1, 0.1, 0.1]
start_x = 1
start_y = 1
goal_offset = [0.5, 0.5, 0.5, 0.5]
goal_x = 2
goal_y = 3
pref_speed = 0.2
radius = 0.1
initial_heading = -np.pi
policies = ['Formation', 'Formation', 'Formation', 'Formation']
policy = policy_dict[policies[0]]
dynamics_model = FullDynamics
agent_list = []
agents = []
for i in range(agent_num):
    agent = MyDict({
        'id': i,
        'start_x': lambda self: start_x + start_offset[self['id']],
        'start_y': lambda self: start_y + start_offset[self['id']],
        'goal_x': lambda self: goal_x + goal_offset[self['id']],
        'goal_y': lambda self: goal_y + goal_offset[self['id']],
        'pref_speed': pref_speed,
        'radius': radius,
        'initial_heading': initial_heading,
        'policy': lambda self: policy_dict[policies[self['id']]],
        'dynamics_model': dynamics_model,
    })
    agent_list.append(agent)
# param_dict = [
#     [MyDict({
#         'id' : 0,
#         'start_x' : lambda self : start_x + start_offset[self['id']],
#         'start_y' : lambda self : start_y + start_offset[self['id']],
#         'goal_x' : lambda self : goal_x + goal_offset[self['id']],
#         'goal_y' : lambda self : goal_y + goal_offset[self['id']],
#         'pref_speed' : pref_speed,
#         'radius' : radius,
#         'initial_heading' : initial_heading,
#         'policy' : lambda self : policy_dict[policies[self['id']]],
#         'dynamics_model' : dynamics_model,
#     })],
#     [MyDict({
#         'id' : 1,
#         'start_x': lambda self: start_x + start_offset[self['id']],
#         'start_y': lambda self: start_y + start_offset[self['id']],
#         'goal_x': lambda self: goal_x + goal_offset[self['id']],
#         'goal_y': lambda self: goal_y + goal_offset[self['id']],
#         'pref_speed': pref_speed,
#         'radius': radius,
#         'initial_heading': initial_heading,
#         'policy' : lambda self : policy_dict[policies[self['id']]],
#         'dynamics_model': dynamics_model,
#     })],
#     [MyDict({
#         'id' : 2,
#         'start_x': lambda self: start_x + start_offset[self['id']],
#         'start_y': lambda self: start_y + start_offset[self['id']],
#         'goal_x': lambda self: goal_x + goal_offset[self['id']],
#         'goal_y': lambda self: goal_y + goal_offset[self['id']],
#         'pref_speed': pref_speed,
#         'radius': radius,
#         'initial_heading': initial_heading,
#         'policy' : lambda self : policy_dict[policies[self['id']]],
#         'dynamics_model': dynamics_model,
#     })],
#     [MyDict({
#         'id' : 3,
#         'start_x': lambda self: start_x + start_offset[self['id']],
#         'start_y': lambda self: start_y + start_offset[self['id']],
#         'goal_x': lambda self: goal_x + goal_offset[self['id']],
#         'goal_y': lambda self: goal_y + goal_offset[self['id']],
#         'pref_speed': pref_speed,
#         'radius': radius,
#         'initial_heading': initial_heading,
#         'policy' : lambda self : policy_dict[policies[self['id']]],
#         'dynamics_model': dynamics_model,
#     })],
# ]


def build_agents():

    for i in range(len(agent_list)):
        Agent(id = agent_list[i]['id'],
              start_x = agent_list[i]['start_x'],
              start_y = agent_list[i]['start_y'],
              goal_x = agent_list[i]['goal_x'],
              goal_y = agent_list[i]['goal_y'],
              pref_speed = agent_list[i]['pref_speed'],
              radius = agent_list[i]['param'],
              initial_heading = agent_list[i]['initial_heading'],
              policy = agent_list[i]['policy'],
              dynamics_model = agent_list[i]['dynamics_model'],
              )
        agents.append(Agent)


# def build_agents(sx, sy, gx, gy, ):
#     s_x = -2
#     s_y = -2
#     # StartList = [[s_x, s_y+1],
#     #              [s_x+1, s_y-1],
#     #              [s_x-1, s_y-1],
#     #              [s_x-1, s_y]]
#     StartList = [[s_x+0.1, s_y+0.1],
#                  [s_x+0.1, s_y-0.1],
#                  [s_x-0.1, s_y-0.1],
#                  [s_x-0.1, s_y+0.1]]
#     goal_x = 2
#     goal_y = 3
#     radius = 0.1
#     pref_speed = 0.2
#     initial_heading = -np.pi
#     policies = ['Formation', 'Formation', 'Formation', 'Formation']
#     agents = [
#         Agent(start_x = StartList[0][0], start_y = StartList[0][1],
#               goal_x = goal_x+0.5, goal_y = goal_y+0.5,
#               neighbor_info={1: [0, -1], 3: [-1, 0]},
#               neighbor_ids = [1,3], neighbor_pos = [[0, -1], [-1, 0]],
#               radius = radius, pref_speed = pref_speed, initial_heading = initial_heading,
#               policy = policy_dict[policies[0]],
#               dynamics_model = FullDynamics,
#               sensors = [OtherAgentsStatesSensor],
#               id = 0),
#         Agent(start_x = StartList[1][0], start_y = StartList[1][1],
#               goal_x = goal_x+0.5, goal_y = goal_y-0.5,
#               neighbor_info={0: [0, 1], 2: [-1, 0]},
#               neighbor_ids = [0,2], neighbor_pos = [[0, 1], [-1, 0]],
#               radius = radius, pref_speed = pref_speed, initial_heading = initial_heading,
#               policy = policy_dict[policies[1]],
#               dynamics_model = FullDynamics,
#               sensors = [OtherAgentsStatesSensor],
#               id = 1),
#         Agent(start_x = StartList[2][0], start_y = StartList[2][1],
#               goal_x = goal_x-0.5, goal_y = goal_y-0.5,
#               neighbor_info={0: [1, 0], 2: [0, 1]},
#               neighbor_ids = [0,2], neighbor_pos = [[1, 0], [0, 1]],
#               radius = radius, pref_speed = pref_speed, initial_heading = initial_heading,
#               policy = policy_dict[policies[2]],
#               dynamics_model = FullDynamics,
#               sensors = [OtherAgentsStatesSensor],
#               id = 2),
#         Agent(start_x = StartList[3][0], start_y = StartList[3][1],
#               goal_x = goal_x-0.5, goal_y = goal_y+0.5,
#               neighbor_info={1: [1, 0], 3: [0, -1]},
#               neighbor_ids = [1,3], neighbor_pos = [[1, 0], [0, -1]],
#               radius = radius, pref_speed = pref_speed, initial_heading = initial_heading,
#               policy = policy_dict[policies[3]],
#               dynamics_model = FullDynamics,
#               sensors = [OtherAgentsStatesSensor],
#               id = 3),
#         # obstacle(start_x=2, start_y=2, radius=0.5,id=4,neighbor_info={1: [1, 0], 3: [0, -1]},
#         #       neighbor_ids = [1,3], neighbor_pos = [[1, 0], [0, -1]],
#         #          dynamics_model = FullDynamics ,sensors = [OtherAgentsStatesSensor] ),
#         obstacle1(start_x=3, start_y=1, radius=0.5, id = 4),
#         ]
#     return agents

def main():
    # Create single tf session for all experiments
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    tf.Session().__enter__()

    # Instantiate the environment
    env = gym.make("Formation-v0")
    # env1 = gym.make("Formation-v0")

    # In case you want to save plots, choose the directory
    env.set_plot_save_dir(os.path.dirname(os.path.realpath(__file__)) + '/results/exam/')
    # env1.set_plot_save_dir(os.path.dirname(os.path.realpath(__file__)) + '/results/exam/')

    # Set agent configuration (start/goal pos, radius, size, policy)
    agents = build_agents()
    [agent.policy.initialize_network() for agent in agents if hasattr(agent.policy, 'initialize_network')]
    env.set_agents(agents)


    obs = env.reset() # Get agents' initial observations

    # Repeatedly send actions to the environment based on agents' observations
    num_steps = 800
    for i in range(num_steps):
        # Query the external agents' policies
        actions = {}

        # e.g., actions[0] = external_policy(dict_obs[0])
        # Internal agents (running a pre-learned policy defined in envs/policies)
        # will automatically query their policy during env.step ==> no need to supply actions for internal agents here

        # Run a simulation step (check for collisions, move sim agents)
        obs, rewards, game_over, which_agents_done = env.step(actions)


        if game_over:
            print("All agents finished!")
            break

    env.reset()

    return True

if __name__ == '__main__':
    main()
    print("Experiment over.")
