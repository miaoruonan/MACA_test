from maca.configs.config import Config
print('with my config')

class EvaluateConfig(Config):
    def __init__(self):
        self.MAX_NUM_AGENTS_IN_ENVIRONMENT = 4
        Config.__init__(self)
        self.EVALUATE_MODE = True
        self.TRAIN_MODE = False
        self.DT = 0.1
        self.MAX_TIME_RATIO = 8.


class Example(EvaluateConfig):
    def __init__(self):
        EvaluateConfig.__init__(self)
        self.DT = 0.01
        self.SAVE_EPISODE_PLOTS = True
        self.PLOT_CIRCLES_ALONG_TRAJ = True
        self.ANIMATE_EPISODES = True
        # self.SENSING_HORIZON = 4
        # self.PLT_LIMITS = [[-20, 20], [-20, 20]]
        # self.PLT_FIG_SIZE = (10,10)


class CADRLTest(EvaluateConfig):
    def __init__(self):
        EvaluateConfig.__init__(self)
        self.SAVE_EPISODE_PLOTS = True
        self.SHOW_EPISODE_PLOTS = False
        self.ANIMATE_EPISODES = True
        self.NEAR_GOAL_THRESHOLD = 0.2
        self.PLT_LIMITS = [[-5, 5], [-5, 5]]
        self.PLT_FIG_SIZE = (12,12)
        self.PLOT_CIRCLES_ALONG_TRAJ = False
        self.NUM_AGENTS_TO_TEST = 6
        self.POLICIES_TO_TEST = ['GA3C-CADRL']



class Formations(EvaluateConfig):
    def __init__(self):
        EvaluateConfig.__init__(self)
        self.SAVE_EPISODE_PLOTS = True
        self.SHOW_EPISODE_PLOTS = False
        self.ANIMATE_EPISODES = True
        self.NEAR_GOAL_THRESHOLD = 0.2
        self.PLT_LIMITS = [[-5, 6], [-2, 7]]
        self.PLT_FIG_SIZE = (10,10)
        self.PLOT_CIRCLES_ALONG_TRAJ = False
        self.NUM_AGENTS_TO_TEST = [6]
        # self.POLICIES_TO_TEST = ['GA3C-CADRL-10']
        self.POLICIES_TO_TEST = ['GA3C-CADRL']
        self.NUM_TEST_CASES = 2
        self.LETTERS = ['C', 'A', 'D', 'R', 'L']

class SmallTestSuite(EvaluateConfig):
    def __init__(self):
        EvaluateConfig.__init__(self)
        self.SAVE_EPISODE_PLOTS = True
        self.SHOW_EPISODE_PLOTS = False
        self.ANIMATE_EPISODES = False
        self.PLOT_CIRCLES_ALONG_TRAJ = True
        self.NUM_TEST_CASES = 4

class FullTestSuite(EvaluateConfig):
    def __init__(self):
        self.MAX_NUM_OTHER_AGENTS_OBSERVED = 19
        EvaluateConfig.__init__(self)
        self.SAVE_EPISODE_PLOTS = True
        self.SHOW_EPISODE_PLOTS = False
        self.ANIMATE_EPISODES = False
        self.PLOT_CIRCLES_ALONG_TRAJ = True

        self.NUM_TEST_CASES = 4
        self.NUM_AGENTS_TO_TEST = [2,3,4]
        self.RECORD_PICKLE_FILES = False

        # # DRLMACA
        # self.FIXED_RADIUS_AND_VPREF = True
        # self.NEAR_GOAL_THRESHOLD = 0.8

        # Normal
        self.POLICIES_TO_TEST = [
            'CADRL', 'RVO', 'GA3C-CADRL-10'
            # 'GA3C-CADRL-4-WS-4-1', 'GA3C-CADRL-4-WS-4-2', 'GA3C-CADRL-4-WS-4-3', 'GA3C-CADRL-4-WS-4-4', 'GA3C-CADRL-4-WS-4-5',
            # 'GA3C-CADRL-4-WS-6-1', 'GA3C-CADRL-4-WS-6-2', 'GA3C-CADRL-4-WS-6-3', 'GA3C-CADRL-4-WS-6-4',
            # 'GA3C-CADRL-4-WS-8-1', 'GA3C-CADRL-4-WS-8-2', 'GA3C-CADRL-4-WS-8-3', 'GA3C-CADRL-4-WS-8-4',
            # 'GA3C-CADRL-4-LSTM-1', 'GA3C-CADRL-4-LSTM-2', 'GA3C-CADRL-4-LSTM-3', 'GA3C-CADRL-4-LSTM-4', 'GA3C-CADRL-4-LSTM-5',
            # 'GA3C-CADRL-10-WS-4-1', 'GA3C-CADRL-10-WS-4-2', 'GA3C-CADRL-10-WS-4-3', 'GA3C-CADRL-10-WS-4-4', 'GA3C-CADRL-10-WS-4-5',
            # 'GA3C-CADRL-10-WS-6-1', 'GA3C-CADRL-10-WS-6-2', 'GA3C-CADRL-10-WS-6-3', 'GA3C-CADRL-10-WS-6-4',
            # 'GA3C-CADRL-10-WS-8-1', 'GA3C-CADRL-10-WS-8-2', 'GA3C-CADRL-10-WS-8-3', 'GA3C-CADRL-10-WS-8-4',
            # 'GA3C-CADRL-10-LSTM-1', 'GA3C-CADRL-10-LSTM-2', 'GA3C-CADRL-10-LSTM-3', 'GA3C-CADRL-10-LSTM-4', 'GA3C-CADRL-10-LSTM-5',
            # 'CADRL', 'RVO'
            ]
        self.FIXED_RADIUS_AND_VPREF = False
        self.NEAR_GOAL_THRESHOLD = 0.2


class CollectRegressionDataset(EvaluateConfig):
    def __init__(self):
        self.MAX_NUM_AGENTS_IN_ENVIRONMENT = 4
        self.MAX_NUM_AGENTS_TO_SIM = 4
        self.DATASET_NAME = ""

        # # Laserscan mode
        # self.USE_STATIC_MAP = True
        # self.STATES_IN_OBS = ['is_learning', 'num_other_agents', 'dist_to_goal', 'heading_ego_frame', 'pref_speed', 'radius', 'laserscan']
        # self.DATASET_NAME = "laserscan_"

        EvaluateConfig.__init__(self)
        self.TEST_CASE_ARGS['policies'] = 'CADRL'
        self.AGENT_SORTING_METHOD = "closest_first"

        # # Laserscan mode
        # self.TEST_CASE_ARGS['agents_sensors'] = ['laserscan', 'other_agents_states']

