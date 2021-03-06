"""
Wrapper for augmented reward
"""
import gym
import numpy as np
from collections import deque
from elastic.envs.atari_wrappers import *
from elastic.envs.video_featurizer import ConvolutionalAutoEncoder
from elastic.utils.common import CheckPeriodic

# from torch import cuda
# cuda.set_device(0)
def debug(*args):
    print('DEBUG', *args)

norm = np.linalg.norm

def _cosine_distance(f1, f2):
    return 1.0 - np.dot(f1, f2) / (norm(f1) * norm(f2))

    
def _l1_distance(f1, f2):
    return norm(f1 - f2, 1) / (norm(f1, 1) * norm(f2, 1))

    
def _l2_distance(f1, f2):
    return norm(f1 - f2) / (norm(f1) * norm(f2))

    
VIDEO_DISTANCE_METRIC = {
    'cosine': _cosine_distance,
    'l1': _l1_distance,
    'l2': _l2_distance,
}


class ClippedRewardsWrapper(gym.Wrapper):
    def _step(self, action):
        obs, reward, done, info = self.env.step(action)
        return obs, np.sign(reward), done, info

debug_ep = CheckPeriodic(5)

class VideoSegmentWrapper(gym.Wrapper):
    def __init__(self, env, 
                 goals, 
                 stack_size,
                 featurizer, 
                 goal_epsilon,
                 distance_metric='cosine', 
                 augmented_reward_weight=1.0):
        """
        
        """
        super(VideoSegmentWrapper, self).__init__(env)
        self.goals = goals
        self.goal_i = 0 # current video segment
        self.obs_stack = deque(maxlen=stack_size)
        self.featurizer = featurizer
        if isinstance(distance_metric, str):
            self.distance_metric = VIDEO_DISTANCE_METRIC[distance_metric.lower()]
        else:
            self.distance_metric = distance_metric
        if isinstance(goal_epsilon, list):
            self.goal_epsilons = goal_epsilon
        else:
            self.goal_epsilons = [goal_epsilon] * len(self.goals)
        self.augmented_reward_weight = augmented_reward_weight
        

    def _step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.obs_stack.append(obs)
        if self.goal_i < len(self.goals):
            cur_goal = self.goals[self.goal_i]
            # shape: stack_size x channel x width x height
            obstack = np.swapaxes(np.stack(self.obs_stack, axis=0), 3, 1)
            cur_distance = self.distance_metric(self.featurizer(obstack), cur_goal)
            if abs(cur_distance) < self.goal_epsilons[self.goal_i]:
                # goal achieved, move to the next goal
                self.goal_i += 1

            reward -= cur_distance[0] * self.augmented_reward_weight
        # TODO: add goal_vector to observation
        return (obs, self.get_goal_vector()), reward, done, info
#         return obs, reward, done, info


    def _reset(self):
        obs = self.env.reset()
        for _ in range(self.obs_stack.maxlen):
            self.obs_stack.append(obs)
        self.goal_i = 0
        return (obs, self.get_goal_vector())
#         return obs
    
    
    def get_goal_vector(self):
        """
        Mark completed stages as 1 and the rest as 0
        """
        vec = np.zeros(len(self.goals))
        vec[:self.goal_i] = 1.0
        return vec


def wrap_video_segment(env, mode, scale_float=True, crop='auto', use_stack=True):
    assert 'NoFrameskip' in env.spec.id
    is_train = (mode == 'train')

    if crop == 'auto':
        NO_CROP_GAMES = [] # not sure whether Qbert should be cropped or not
        crop = not any((game in env.spec.id) for game in NO_CROP_GAMES)
        print('wrap_video_segment: crop =', crop)
    if is_train:
        env = EpisodicLifeEnv(env)
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4, max=True)
    env = FireResetEnv(env)
    if is_train:
        env = ClippedRewardsWrapper(env)
    featurizer = ConvolutionalAutoEncoder('seaquest_video_info.pickle')
    goals, epsilons = featurizer.get_goals_and_variances()
    if is_train:
        env = VideoSegmentWrapper(env, 
                                  goals=goals,
                                  stack_size=5,
                                  featurizer=featurizer.get_latent,
                                  goal_epsilon=epsilons)
    env = ProcessFrame84(env, crop=crop)
    if use_stack:
        env = StackFrameWrapper(env, buff=4)
    if scale_float:
        env = RescaleFrameFloat(env)
    return env


class FilterWrapper(gym.Wrapper):
    def __init__(self, env, wrapper):
        """
        Stack the last n frames as input channels

        Args:
          buff: number of last frames to be stacked
        """
        super(FilterWrapper, self).__init__(env)
        self.wrapper = wrapper


    def _step(self, action):
        obs, reward, done, info = self.env.step(action)
        assert isinstance(obs, tuple) and len(obs) == 2
        aug = obs[1]
        obs = obs[0]
        pass


    def _reset(self):
        pass
    

def wrap_video_segment_with_goal(env, mode, scale_float=True, crop='auto', use_stack=True):
    assert 'NoFrameskip' in env.spec.id
    is_train = (mode == 'train')

    if crop == 'auto':
        NO_CROP_GAMES = [] # not sure whether Qbert should be cropped or not
        crop = not any((game in env.spec.id) for game in NO_CROP_GAMES)
        print('wrap_video_segment: crop =', crop)
    if is_train:
        env = EpisodicLifeEnv(env)
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4, max=True)
    env = FireResetEnv(env)
    if is_train:
        env = ClippedRewardsWrapper(env)
    featurizer = ConvolutionalAutoEncoder('seaquest_video_info.pickle')
    goals, epsilons = featurizer.get_goals_and_variances()
    if is_train:
        env = VideoSegmentWrapper(env, 
                                  goals=goals,
                                  stack_size=5,
                                  featurizer=featurizer.get_latent,
                                  goal_epsilon=epsilons)
    env = ProcessFrame84(env, crop=crop)
    if use_stack:
        env = StackFrameWrapper(env, buff=4)
    if scale_float:
        env = RescaleFrameFloat(env)
    return env


if __name__ == '__main__':
    import sys
    import matplotlib.pyplot as plt
    if len(sys.argv) < 2:
        env_id = 'SpaceInvaders'
    else:
        env_id = sys.argv[1]
    env = gym.make('{}NoFrameskip-v3'.format(env_id))
    A = np.array
    goals = [(1,1), (0,1), (-1,0), (0,-1), (1, 0)]
    goals = list(map(np.array, goals))
    step = 0
    def featurizer(obs):
        global step
        theta = step / 1500.0 * 2*np.pi
        return np.array([np.cos(theta), np.sin(theta)])
    env = ClippedRewardsWrapper(env)
    env = VideoSegmentWrapper(env, 
                              goals=goals, 
                              stack_size=1, 
                              featurizer=featurizer, 
                              goal_epsilon=1e-5)
    env.reset()
    reward_history = []
    goal_vec_history = []
    while True:
        obs, r, done, info = env.step(env.action_space.sample())
        step += 1
        reward_history.append(r)
        goal_vec_history.append(np.count_nonzero(obs[1]))
        if done:
            break
    print('total steps', step)
    goal_vec_history = np.array(goal_vec_history, np.float32) / len(obs[1])
    plt.plot(reward_history)
    plt.plot(goal_vec_history)
    plt.axis([0, 1900, -1, 1])
    plt.show()
