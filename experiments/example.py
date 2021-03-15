import os
import gym
import numpy as np
gym.logger.set_level(40)
import tensorflow as tf

os.environ['GYM_CONFIG_CLASS'] = 'Example'

from maca.agents.agent import Agent
# Policies
from maca.policies import StaticPolicy
from maca.policies import NonCooperativePolicy
# from maca.envs.policies.DRLLongPolicy import DRLLongPolicy
from maca.policies.RVOPolicy import RVOPolicy
from maca.policies import CADRLPolicy
from maca.policies import GA3CCADRLPolicy
from maca.policies import ExternalPolicy
from maca.policies import LearningPolicy
from maca.policies import CARRLPolicy
from maca.policies.LearningPolicyGA3C import LearningPolicyGA3C
# Dynamics
from maca.envs.dynamics.UnicycleDynamics import UnicycleDynamics
# Sensors
from maca.envs.sensors.OtherAgentsStatesSensor import OtherAgentsStatesSensor

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
}


def build_agents():
    goal_x = 3
    goal_y = 3
    radius = 0.1
    pref_speed = 1.0
    initial_heading = np.pi
    policies = ['GA3C_CADRL', 'GA3C_CADRL', 'GA3C_CADRL', 'GA3C_CADRL']
    agents = [
        Agent(start_x=-goal_x, start_y=-goal_y, goal_x = goal_x, goal_y = goal_y,
              radius = radius, pref_speed = pref_speed, initial_heading = initial_heading,
              policy = policy_dict[policies[0]],
              dynamics_model = UnicycleDynamics,
              sensors = [OtherAgentsStatesSensor],
              id = 0),
        Agent(start_x=-goal_x, start_y=goal_y, goal_x = goal_x, goal_y = -goal_y,
              radius = radius, pref_speed = pref_speed, initial_heading = initial_heading,
              policy = policy_dict[policies[1]],
              dynamics_model = UnicycleDynamics,
              sensors = [OtherAgentsStatesSensor],
              id = 1),
        Agent(start_x=goal_x, start_y=goal_y, goal_x = -goal_x, goal_y = -goal_y,
              radius = radius, pref_speed = pref_speed, initial_heading = initial_heading,
              policy = policy_dict[policies[2]],
              dynamics_model = UnicycleDynamics,
              sensors = [OtherAgentsStatesSensor],
              id = 2),
        Agent(start_x=goal_x, start_y=-goal_y, goal_x = -goal_x, goal_y = goal_y,
              radius = radius, pref_speed = pref_speed, initial_heading = initial_heading,
              policy = policy_dict[policies[3]],
              dynamics_model = UnicycleDynamics,
              sensors = [OtherAgentsStatesSensor],
              id = 3)
        ]
    return agents

def main():
    # Create single tf session for all experiments
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    tf.Session().__enter__()

    # Instantiate the environment
    env = gym.make("CollisionAvoidance-v0")

    # In case you want to save plots, choose the directory
    env.set_plot_save_dir(os.path.dirname(os.path.realpath(__file__)) + '/results/example/')

    # Set agent configuration (start/goal pos, radius, size, policy)
    agents = build_agents()
    [agent.policy.initialize_network() for agent in agents if hasattr(agent.policy, 'initialize_network')]
    env.set_agents(agents)

    obs = env.reset() # Get agents' initial observations
    # Repeatedly send actions to the environment based on agents' observations
    num_steps = 100
    for i in range(num_steps):
        # Query the external agents' policies
        # e.g., actions[0] = external_policy(dict_obs[0])
        actions = {}
        actions[0] = np.array([1., 0.5])

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
