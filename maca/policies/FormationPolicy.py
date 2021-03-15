import os
import numpy as np
from maca.envs import util
from maca.envs import Config
from maca.envs.util import wrap
from maca.policies.InternalPolicy import InternalPolicy
from maca.policies.CADRL.scripts.multi import nn_navigation_value_multi as nn_nav

class FormationPolicy(InternalPolicy):
    """ Re-purposed from: Socially Aware Motion Planning with Deep Reinforcement Learning

    Loads a pre-traned SA-CADRL 4-agent network (with no social norm preference LHS/RHS).
    Some methods to convert the gym agent representation to the numpy arrays used in the old code.

    """
    def __init__(self):
        InternalPolicy.__init__(self, str="RULE")
        num_agents = 4
        # load value_net
        # mode = 'rotate_constr'; passing_side = 'right'; iteration = 1300
        mode = 'no_constr'
        passing_side = 'none'
        iteration = 1000
        self.w_p = 3
        self.w_g = 1

    def find_next_action(self, obs, agents, index):
        """
        Converts environment's agents representation to CADRL format.
        Go at pref_speed, apply a change in heading equal to zero out current ego heading (heading to goal)
        Args:
            obs (dict): ignored
            agents (list): of Agent objects
            index (int): this agent's index in that list
        Returns:
            np array of shape (2,)... [spd, delta_heading]
            commanded [heading delta, speed]
        """
        host_agent   = agents[index]
        neighbor_num = len(host_agent.neighbor_info)
        pos_diff_sum = 0
        vel_diff_sum = 0
        for idx, pos_diff in host_agent.neighbor_info.items():
            other_agent = agents[idx]
            rel_pos_to_other_global_frame = other_agent.pos_global_frame - host_agent.pos_global_frame
            pos_diff_sum += rel_pos_to_other_global_frame - pos_diff
            vel_diff_sum += other_agent.vel_global_frame #- host_agent.vel_global_frame

        vel_consensus = np.array(self.w_p * pos_diff_sum + 1/neighbor_num * vel_diff_sum )

        if index == 0:
            selected_speed   = host_agent.pref_speed
            selected_heading = wrap(-host_agent.heading_ego_frame + host_agent.heading_global_frame)
            vel_goal = selected_speed * np.array([np.cos(selected_heading), np.sin(selected_heading)])
            action = vel_consensus + self.w_g * vel_goal
        else:
            action = vel_consensus

        return action

    def convert_host_agent_to_cadrl_state(self, agent):
        """ Convert this repo's state representation format into the legacy cadrl format for the host agent
        Args:
            agent (:class:`~gym_collision_avoidance.envs.agent.Agent`): this agent
        Returns:
            10-element (np array) describing current state
        """

        # rel pos, rel vel, size
        x = agent.pos_global_frame[0]; y = agent.pos_global_frame[1]
        v_x = agent.vel_global_frame[0]; v_y = agent.vel_global_frame[1]
        radius = agent.radius; turning_dir = agent.turning_dir
        heading_angle = agent.heading_global_frame
        pref_speed = agent.pref_speed
        goal_x = agent.goal_global_frame[0]; goal_y = agent.goal_global_frame[1]
        
        agent_state = np.array([x, y, v_x, v_y,
                                heading_angle, pref_speed,
                                goal_x, goal_y, radius, turning_dir])

        return agent_state

    def convert_other_agents_to_cadrl_state(self, host_agent, other_agents):
        """ Convert this repo's state representation format into the legacy cadrl format for the other agents in the environment.
        Args:
            host_agent (:class:`~envs.agent.Agent`): this agent
            other_agents (list): of all the other :class:`~envs.agent.Agent` objects
        Returns:
            - (3 x 10) np array (this cadrl can handle 3 other agents), each has 10-element state vector
        """
        other_agent_dists = []
        for i, other_agent in enumerate(other_agents):
            # project other elements onto the new reference frame
            rel_pos_to_other_global_frame = other_agent.pos_global_frame - host_agent.pos_global_frame
            p_orthog_ego_frame = np.dot(rel_pos_to_other_global_frame, host_agent.ref_orth)
            dist_between_agent_centers = np.linalg.norm(rel_pos_to_other_global_frame)
            dist_2_other = dist_between_agent_centers - host_agent.radius - other_agent.radius
            other_agent_dists.append([i,round(dist_2_other,2), p_orthog_ego_frame])

        sorted_dists = sorted(other_agent_dists, key = lambda x: (-x[1], x[2]))
        sorted_inds = [x[0] for x in sorted_dists]
        clipped_sorted_inds = sorted_inds[-min(Config.MAX_NUM_OTHER_AGENTS_OBSERVED,3):]
        clipped_sorted_agents = [other_agents[i] for i in clipped_sorted_inds]

        other_agents_state = []
        for agent in clipped_sorted_agents:
            x      = agent.pos_global_frame[0]; y = agent.pos_global_frame[1]
            v_x    = agent.vel_global_frame[0]; v_y = agent.vel_global_frame[1]
            radius = agent.radius
            turning_dir = agent.turning_dir
            heading_angle = agent.heading_global_frame
            pref_speed = agent.pref_speed
            goal_x = agent.goal_global_frame[0]
            goal_y = agent.goal_global_frame[1]

            other_agent_state = np.array([x, y, v_x, v_y, heading_angle, pref_speed, goal_x, goal_y, radius, turning_dir])
            other_agents_state.append(other_agent_state)
        return other_agents_state