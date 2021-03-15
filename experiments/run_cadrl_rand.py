import os
import numpy as np

os.environ['GYM_CONFIG_CLASS'] = 'CADRLTest'
os.environ['GYM_CONFIG_PATH'] = '../maca/configs/my_config.py'


from maca.envs import Config
from maca.envs.test_cases import get_testcase_random
from maca.env_utils import create_env, policy_dict


if __name__ == '__main__':

    np.random.seed(0)

    # num_agents = Config.NUM_AGENTS_TO_TEST
    policy = Config.POLICIES_TO_TEST[0]
    policy_cfg = policy_dict[policy]

    agents = get_testcase_random(num_agents=4,
                                 policies = [policy_cfg['policy'], policy_cfg['policy'], policy_cfg['policy']],
                                 policy_distr = [0.05, 0.9, 0.05],
                                 speed_bnds = [0.5, 2.0],
                                 radius_bnds = [0.2, 0.8],
                                 side_length=[{'num_agents': [0, 5], 'side_length': [4, 5]},
                                              {'num_agents': [5, np.inf], 'side_length': [6, 8]},]
                                 )

    env, one_env = create_env()
    one_env.set_agents(agents)
    one_env.plot_policy_name = policy
    one_env.set_plot_save_dir(os.path.dirname(os.path.realpath(__file__)) + '/results/cadrl_rand_case/')

    for agent in agents:
        if 'checkpt_name' in policy_cfg:
            agent.policy.env = env
            agent.policy.initialize_network(**policy_cfg)
        if 'sensor_args' in policy_cfg:
            for sensor in agent.sensors:
                sensor.set_args(policy_cfg['sensor_args'])

    env.reset()
    total_reward = 0
    step = 0
    done = False
    while not done:
        obs, rew, done, info = env.step([None])
        total_reward += rew[0]
        step += 1

    print("Experiment over with total_reward: ", total_reward, ' in ', step, 'steps.')
