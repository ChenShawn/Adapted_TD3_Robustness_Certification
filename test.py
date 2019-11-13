import gym
import pybullet_envs
from TD3 import TD3
from PIL import Image
import argparse
import numpy as np
import torch
import copy


def parse_arguments():
    parser = argparse.ArgumentParser("TRAINING")
    parser.add_argument('-e', "--env", type=str, default="LunarLanderContinuous-v2", help="env name")
    parser.add_argument('-n', "--n-episodes", type=int, default=10, help="number of episodes")
    parser.add_argument('-m', "--relative-mass", type=float, default=1.0, help="relative-mass")
    parser.add_argument("--train-seed", type=int, default=1, help="random seed for training")
    parser.add_argument("--test-seed", type=int, default=1, help="random seed for testing")
    parser.add_argument("--render", action="store_true", default=False)
    parser.add_argument("--save-gif", action="store_true", default=False)
    parser.add_argument("--ensemble", action="store_true", default=False)
    parser.add_argument("--multiple", action="store_true", default=False)
    return parser.parse_args()


def gen_envs(arglist):
    env = gym.make(arglist.env)
    if 'model' in dir(env.env):
        # For mujoco envs
        ori_mass = copy.deepcopy(env.env.model.body_mass.copy())
        for idx in range(len(ori_mass)):
            env.env.model.body_mass[idx] = ori_mass[idx] * arglist.relative_mass
    elif 'world' in dir(env.env):
        # For some of the classic control envs
        env.env.world.gravity *= arglist.relative_mass
    return env


def test(arglist):
    env_name = arglist.env
    random_seed = arglist.test_seed
    n_episodes = arglist.n_episodes
    lr = 0.002
    max_timesteps = 3000
    render = arglist.render
    save_gif = arglist.save_gif
    
    if not arglist.ensemble:
        filename = "TD3_{}_{}".format(env_name, arglist.train_seed)
        # filename += '_solved'
        directory = "./preTrained/{}".format(env_name)
    else:
        filename = "TD3_{}_{}_ensemble".format(env_name, arglist.train_seed)
        # filename += '_solved'
        directory = "./preTrained/{}".format(env_name)
    
    #env = gym.make(env_name)
    env = gen_envs(arglist)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    if random_seed:
        env.seed(random_seed)
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
    
    policy = TD3(lr, state_dim, action_dim, max_action)
    policy.load_actor(directory, filename)
    
    total_reward_list = []
    for ep in range(1, n_episodes+1):
        ep_reward = 0.0
        state = env.reset()
        for t in range(max_timesteps):
            action = policy.select_action(state)
            state, reward, done, _ = env.step(action)
            ep_reward += reward
            if render:
                env.render()
                if save_gif:
                     img = env.render(mode = 'rgb_array')
                     img = Image.fromarray(img)
                     img.save('./gif/{}.jpg'.format(t))
            if done:
                break
            
        print('Episode: {}\tReward: {}'.format(ep, int(ep_reward)))
        total_reward_list.append(ep_reward)
        ep_reward = 0.0
    env.close()
    return total_reward_list


def multi_test(arglist):
    """multi_test
        Testing performances under various random seeds and relative mass values
        return: a list of dicts [{env, random_seed, relative_mass, mean, half_std}, {}, ...]
    """
    import gc
    relative_mass_list = np.arange(0.5, 1.501, 0.025).tolist()
    # relative_mass_list = [0.9, 1.0, 1.1]
    log_filename = './logs/TD3_{}_{}_robust.log'.format(arglist.env, arglist.test_seed)

    args = parse_arguments()
    result_list = []
    log_fd = open(log_filename, 'w+')
    for mass in relative_mass_list:
        arglist.relative_mass = mass
        reward_list = test(arglist)

        reward_array = np.array(reward_list, dtype=np.float32)
        reward_mean = reward_array.mean()
        reward_half_std = reward_array.std() / 2.0
        loginfo = 'env={}, random_seed={}, relative_mass={}, result={}±{}'
        print(loginfo.format(args.env, args.test_seed, mass, reward_mean, reward_half_std))

        info_dict = {'env': arglist.env, 'random_seed': arglist.test_seed, 'relative_mass': mass, 
                     'reward_mean': reward_mean, 'reward_half_std': reward_half_std}
        result_list.append(info_dict)
        log_fd.write(loginfo.format(args.env, args.test_seed, mass, reward_mean, reward_half_std) + '\n')
        log_fd.flush()
        gc.collect()
    log_fd.close()
    return result_list
    


if __name__ == '__main__':
    args = parse_arguments()

    if not args.multiple:
        reward_list = test(args)

        reward_array = np.array(reward_list, dtype=np.float32)
        reward_mean = reward_array.mean()
        reward_half_std = reward_array.std() / 2.0
        loginfo = ' env={}, random_seed={}, relative_mass={}, result={}±{}'
        print(loginfo.format(args.env, args.test_seed, args.relative_mass, reward_mean, reward_half_std))

    else:
        result_list = multi_test(args)
        xs = [info['relative_mass'] for info in result_list]
        ys = [info['reward_mean'] for info in result_list]
        ys_half_std = [info['reward_half_std'] for info in result_list]
        ys_lower = np.array(ys, dtype=np.float32) - np.array(ys_half_std, dtype=np.float32)
        ys_upper = np.array(ys, dtype=np.float32) + np.array(ys_half_std, dtype=np.float32)

        import matplotlib.pyplot as plt
        plt.style.use('ggplot')
        plt.figure()
        plt.plot(xs, ys, linewidth=1.5, color='blue')
        plt.fill_between(xs, ys_lower, ys_upper, color='blue', alpha=0.2)
        plt.xlabel('relative mass')
        plt.ylabel('average returns')
        plt.title('Performance under different relative mass values')
        plt.show()
