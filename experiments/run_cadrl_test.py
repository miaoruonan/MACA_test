import os
import numpy as np

os.environ['GYM_CONFIG_CLASS'] = 'CADRLTest'
os.environ['GYM_CONFIG_PATH'] = '../maca/configs/my_config.py'


from maca.envs import Config
from maca.envs.test_cases import cadrl_test_case_to_agents
from maca.env_utils import create_env, policy_dict


if __name__ == '__main__':

    np.random.seed(0)

    num_agents = Config.NUM_AGENTS_TO_TEST
    policy = Config.POLICIES_TO_TEST[0]
    policy_cfg = policy_dict[policy]

    circle_r = 4
    tc = np.zeros((num_agents, 6))
    for i in range(num_agents):
        theta_start = (2 * np.pi / num_agents) * i
        theta_end = theta_start + np.pi
        tc[i, 0] = circle_r * np.cos(theta_start)
        tc[i, 1] = circle_r * np.sin(theta_start)
        tc[i, 2] = circle_r * np.cos(theta_end)
        tc[i, 3] = circle_r * np.sin(theta_end)
        tc[i, 4] = 1.0
        tc[i, 5] = 0.1

    agents = cadrl_test_case_to_agents(test_case=tc,
                                       policies=policy_cfg['policy'],
                                       agents_dynamics='unicycle',
                                       agents_sensors=policy_cfg['sensors'])

    env, one_env = create_env()
    one_env.set_agents(agents)
    one_env.plot_policy_name = policy
    one_env.set_plot_save_dir(os.path.dirname(os.path.realpath(__file__)) + '/results/cadrl_test_case/')

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
