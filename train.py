import torch
import gym
import numpy as np
from TD3 import TD3
from utils import ReplayBuffer
import argparse
import random, copy

def parse_arguments():
    parser = argparse.ArgumentParser("TRAINING")
    parser.add_argument("--env", type=str, default="LunarLanderContinuous-v2", help="name of the env")
    parser.add_argument("--seed", type=int, default=1, help="random-seed")
    # parameters
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    parser.add_argument("--tau", type=float, default=0.01, help="exponential moving average ratio")
    # train
    parser.add_argument("--batch-size", type=int, default=256, help="batch size")
    parser.add_argument("--max-episodes", type=int, default=1000, help="maximum training episodes")
    parser.add_argument("--save-rate", type=int, default=100, help="save policy each...")
    parser.add_argument("--render", action="store_true", default=False)
    parser.add_argument("--ensemble", action="store_true", default=False)
    return parser.parse_args()


def is_solved(env, avg_reward):
    if env == "LunarLanderContinuous-v2":
        return avg_reward > 200
    elif env == 'BipedalWalker-v2' or env == 'BipedalWalkerHardcore-v2':
        return avg_reward > 300
    elif env == 'Ant-v2':
        return avg_reward > 3000
    else:
        return False


def get_random_envs(env_name, lower=0.75, upper=1.25):
    rand = random.random()
    rand = rand * (upper - lower) + lower
    env = gym.make(env_name)
    if 'model' in dir(env.env):                     # for mujoco envs
        ori_mass = copy.deepcopy(env.env.model.body_mass.copy())
        for idx in range(len(ori_mass)):
            env.env.model.body_mass[idx] = ori_mass[idx] * rand
    elif 'world' in dir(env.env):
        env.env.world.gravity *= rand
    return env


def train(arglist):
    ######### Hyperparameters #########
    #env_name = "BipedalWalker-v2"
    env_name = arglist.env
    log_interval = 10                               # print avg reward after interval
    random_seed = arglist.seed                      # only works when random_seed has non-zero value
    gamma = 0.99                                    # discount for future rewards
    batch_size = arglist.batch_size                 # num of transitions sampled from replay buffer
    lr = arglist.lr
    exploration_noise = 0.1 
    polyak = 1.0 - arglist.tau                      # target policy update parameter (1-tau)
    policy_noise = 0.2                              # target policy smoothing noise
    noise_clip = 0.5
    policy_delay = 2                                # delayed policy updates parameter
    max_episodes = arglist.max_episodes             # max num of episodes
    max_timesteps = 2000                            # max timesteps in one episode
    directory = "./preTrained/{}".format(env_name)  # save trained models
    filename = "TD3_{}_{}".format(env_name, random_seed)
    log_filename = './logs/TD3_{}_{}.log'.format(env_name, random_seed)
    ###################################
    
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    
    policy = TD3(lr, state_dim, action_dim, max_action)
    replay_buffer = ReplayBuffer()
    
    if random_seed:
        print("Random Seed: {}".format(random_seed))
        env.seed(random_seed)
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
    
    # logging variables:
    avg_reward = 0
    ep_reward = 0
    log_f = open(log_filename, "w+")
    
    # training procedure:
    for episode in range(1, max_episodes+1):
        state = env.reset()
        for t in range(max_timesteps):
            # select action and add exploration noise:
            action = policy.select_action(state)
            action = action + np.random.normal(0, exploration_noise, size=env.action_space.shape[0])
            action = action.clip(env.action_space.low, env.action_space.high)
            
            # take action in env:
            next_state, reward, done, _ = env.step(action)
            replay_buffer.add((state, action, reward, next_state, float(done)))
            state = next_state
            
            avg_reward += reward
            ep_reward += reward
            
            # if episode is done then update policy:
            if done or t==(max_timesteps-1):
                policy.update(replay_buffer, t, batch_size, gamma, polyak, policy_noise, noise_clip, policy_delay)
                if arglist.ensemble:
                    env.close()
                    env = get_random_envs(env_name)
                break
        
        # logging updates:
        log_f.write('{},{}\n'.format(episode, ep_reward))
        log_f.flush()
        ep_reward = 0
        
        # if avg reward > 300 then save and stop traning:
        # if (avg_reward/log_interval) >= 300:
        if is_solved(env_name, avg_reward / log_interval):
            print("########## SOLVED IN {} EPISODES! ###########".format(episode))
            name = filename + '_solved'
            policy.save(directory, name)
            log_f.close()
            break
        
        if episode > 500 and episode % arglist.save_rate == 0:
            policy.save(directory, filename)
        
        # print avg reward every log interval:
        if episode % log_interval == 0:
            avg_reward = int(avg_reward / log_interval)
            print("Episode: {}\tAverage Reward: {}".format(episode, avg_reward))
            avg_reward = 0
    log_f.close()


if __name__ == '__main__':
    args = parse_arguments()
    train(args)
    
