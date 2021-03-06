import os
import gym
import numpy as np
from maca.configs.config_test import Config
gym.logger.set_level(40)
import tensorflow as tf

os.environ['GYM_CONFIG_CLASS'] = 'Example'
os.environ['GYM_CONFIG_PATH'] = '../maca/configs/my_config.py'

from maca.agents.agent_test import DistributedAgent as Agent
from maca.agents.obstacle1 import Obstacle as Obstacle1
# Policies
from maca.policies import FormationPolicy_test
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
    'Formation_t': FormationPolicy_test,
}

config = Config()
agent_num = 4
# x,y
start_offset = [[0.1, 0.1], [0.1, -0.1], [-0.1, -0.1], [-0.1, 0.1]]
start = [1.0, 1.0]
# x,y
goal_offset = [[0.5, 0.5], [0.5, -0.5], [-0.5, -0.5], [-0.5, 0.5]]
goal = [2.0, 3.0]
pref_speed = 0.1
radius = 0.1
initial_heading = -np.pi
# policies = ['Formation', 'Formation', 'Formation', 'Formation']
policies = ['Formation_t', 'Formation', 'external', 'learning']
policy = policy_dict[policies[0]]
dynamics_model = FullDynamics
sensors = [OtherAgentsStatesSensor]
agent_list = []
agents = []

# neighbor_info = [{1: [0, -1], 3: [-1, 0]}, {0: [0, 1], 2: [-1, 0]}, {0: [1, 0], 2: [0, 1]}, {1: [1, 0], 3: [0, -1]}]
neighbor_info = [{1: [1, 0], 3: [0, 0]}, {0: [0, 0], 2: [1, 0]}, {0: [0, 0], 2: [0, 0]}, {1: [1, 0], 3: [0, 0]}]

agent_dict = {
    'start': start,
    # 'start_y': start,
    'goal': goal,
    # 'goal_y': goal,
    'pref_speed': pref_speed,
    'radius': radius,
    'initial_heading': initial_heading,
    'dynamics_model': dynamics_model,
    'sensors': sensors,
}


def build_agents():
    for i in range(agent_num):
        print(i)
        agent = Agent(id = i,
                      start_x = agent_dict['start'][0] + start_offset[i][0],
                      start_y = agent_dict['start'][1] + start_offset[i][1],
                      goal_x = agent_dict['goal'][0] + goal_offset[i][0],
                      goal_y = agent_dict['goal'][1] + goal_offset[i][1],
                      pref_speed = agent_dict['pref_speed'],
                      radius = agent_dict['radius'],
                      initial_heading = agent_dict['initial_heading'],
                      policy = policy_dict[policies[0]],
                      dynamics_model = FullDynamics,
                      sensors = [OtherAgentsStatesSensor],
                      # neighbor_info = neighbor_info[i],
                      neighbor_info = config.adj_matrix2['test1'][i],
                      )
        print(agent.neighbor_info)
        agents.append(agent)
        # print(agents)
    return agents

# def build_obstacl():
#     obstacle = Obstacle1(2, 2, 0.4, 4)
#     agents.append(obstacle)




def main():
    # Create single tf session for all experiments
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    tf.Session().__enter__()

    # Instantiate the environment
    env = gym.make("Formation-v0")

    # In case you want to save plots, choose the directory
    env.set_plot_save_dir(os.path.dirname(os.path.realpath(__file__)) + '/results/exam/')

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
