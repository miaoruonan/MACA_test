import math
import numpy as np
from maca.envs import Config
from maca.envs.util import wrap


class BaseAgent(object):

	def __deepcopy__(self, memo):
		""" Copy every attribute about the agent except its policy (since that may contain MBs of DNN weights) """
		cls = self.__class__
		obj = cls.__new__(cls)
		for k, v in self.__dict__.items():
			if k != 'policy':
				setattr(obj, k, v)
		return obj


	def get_agent_data(self, attribute):
		""" Grab the value of self.attribute (useful to define which states sensor uses from config file).
		Args:
			attribute (str): which attribute of this agent to look up (e.g., "pos_global_frame")
		"""
		return getattr(self, attribute)

	def get_agent_data_equiv(self, attribute, value):
		""" Grab the value of self.attribute and return whether it's equal to value (useful to define states sensor uses from config file).
		Args:
			attribute (str): which attribute of this agent to look up (e.g., "radius")
			value (anything): thing to compare self.attribute to (e.g., 0.23)

		Returns:
			result of self.attribute and value comparison (bool)
		"""
		return eval("self." + attribute) == value

	def get_observation_dict(self, agents):
		observation = {}
		for state in Config.STATES_IN_OBS:
			observation[state] = np.array(eval("self." + Config.STATE_INFO_DICT[state]['attr']))
		return observation

	def ego_pos_to_global_pos(self, ego_pos):
		""" Convert a position in the ego frame to the global frame.

		This might be useful for plotting some of the perturbation stuff.

		Args:
			ego_pos (np array): if (2,), it represents one (x,y) position in ego frame
				if (n,2), it represents n (x,y) positions in ego frame

		Returns:
			global_pos (np array): either (2,) (x,y) position in global frame or (n,2) n (x,y) positions in global frame

		"""
		if ego_pos.ndim == 1:
			ego_pos_ = np.array([ego_pos[0], ego_pos[1], 1])
			global_pos = np.dot(self.T_global_ego, ego_pos_)
			return global_pos[:2]
		else:
			ego_pos_ = np.hstack([ego_pos, np.ones((ego_pos.shape[0], 1))])
			global_pos = np.dot(self.T_global_ego, ego_pos_.T).T
			return global_pos[:, :2]

	def global_pos_to_ego_pos(self, global_pos):
		""" Convert a position in the global frame to the ego frame.

		Args:
			global_pos (np array): one (x,y) position in global frame

		Returns:
			ego_pos (np array): (2,) (x,y) position in ego frame

		"""
		ego_pos = np.dot(np.linalg.inv(self.T_global_ego), np.array([global_pos[0], global_pos[1], 1]))
		return ego_pos[:2]

