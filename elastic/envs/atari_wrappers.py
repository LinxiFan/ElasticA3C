import os
import cv2
import numpy as np
from collections import deque
import gym
from gym import spaces
from elastic.utils.image import *


class NoopResetEnv(gym.Wrapper):
    def __init__(self, env=None, noop_max=30):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        super(NoopResetEnv, self).__init__(env)
        self.noop_max = noop_max
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def _reset(self):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset()
        noops = np.random.randint(1, self.noop_max + 1)
        for _ in range(noops):
            obs, _, _, _ = self.env.step(0)
        return obs


class FireResetEnv(gym.Wrapper):
    def __init__(self, env=None):
        """Take action on reset for environments that are fixed until firing."""
        super(FireResetEnv, self).__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def _reset(self):
        self.env.reset()
        obs, _, _, _ = self.env.step(1)
        obs, _, _, _ = self.env.step(2)
        return obs


class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env=None):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        super(EpisodicLifeEnv, self).__init__(env)
        self.lives = 0
        self.was_real_done  = True
        self.was_real_reset = False

    def _step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # for Qbert somtimes we stay in lives == 0 condtion for a few frames
            # so its important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def _reset(self):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset()
            self.was_real_reset = True
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
            self.was_real_reset = False
        self.lives = self.env.unwrapped.ale.lives()
        return obs


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env=None, skip=4, max=False):
        """Return only every `skip`-th frame"""
        super(MaxAndSkipEnv, self).__init__(env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = deque(maxlen=2)
        self._skip = skip
        self._max = max


    def _step(self, action):
        total_reward = 0.0
        done = None
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if done:
                break

        if self._max:
            frame = np.max(np.stack(self._obs_buffer), axis=0)
        else:
            frame = self._obs_buffer[-1]
        return frame, total_reward, done, info


    def _reset(self):
        """Clear past frame buffer and init. to first obs. from inner env."""
        self._obs_buffer.clear()
        obs = self.env.reset()
        self._obs_buffer.append(obs)
        return obs


class StackFrameWrapper(gym.Wrapper):
    def __init__(self, env, buff=4):
        """
        Stack the last n frames as input channels

        Args:
          buff: number of last frames to be stacked
        """
        super(StackFrameWrapper, self).__init__(env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = deque(maxlen=buff)
        self._buff = buff
        old_shape = self.observation_space.shape
        assert old_shape[-1] == 1, 'must be single channel'
        new_shape = old_shape[:2] + (buff,)
        self.observation_space = spaces.Box(low=0, high=255, shape=new_shape)


    def _step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._obs_buffer.append(obs)
        return self._stack(), reward, done, info


    def _reset(self):
        """Clear past frame buffer and init. to first obs. from inner env."""
        self._obs_buffer.clear()
        obs = self.env.reset()
        # at the beginning, we fill the buffer with the first frame
        for _ in range(self._buff):
            self._obs_buffer.append(obs)
        return self._stack()
    
    def _stack(self):
        return np.concatenate(self._obs_buffer, axis=-1)


def _process_frame84_old(frame):
    img = np.reshape(frame, [210, 160, 3]).astype(np.float32)
    img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
    resized_screen = cv2.resize(img, (84, 110),  interpolation=cv2.INTER_LINEAR)
    x_t = resized_screen[18:102, :]
    x_t = np.reshape(x_t, [84, 84, 1])
    return x_t.astype(np.uint8)


def _process_frame84(frame, crop=True):
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    if crop:
        frame = cv2.resize(frame, (84, 110),  interpolation=cv2.INTER_AREA)
        return frame[18:102, :, np.newaxis]
    else:
        frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)
        return frame[:, :, np.newaxis]


_DEBUG_STEP = 0
def _process_frame84_debug(frame, crop=True):
    global _DEBUG_STEP
    _DEBUG_STEP += 1
    _debug = _DEBUG_STEP % 100 == 1
    F = os.path.expanduser('~/Temp/imgs/{}.png').format
    show = lambda name: os.system('open ' + F(name))

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    if True or crop:
        frame = cv2.resize(frame, (84, 110),  interpolation=cv2.INTER_AREA)
        if _debug:
            save_img(frame, F('original'))
            show('original')
        frame = frame[18:102, :, np.newaxis]
        if _debug:
            save_img(frame, F('cropped'))
            show('cropped')
            input()
        return frame
    else:
        frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)
        return frame[:, :, np.newaxis]

# DEBUG
# _process_frame84 = _process_frame84_debug


class ProcessFrame84(gym.Wrapper):
    """
    Make sure you rescale to float!
    tf.cast(state, tf.float32) / 255.0
    Rescale on GPU instead of CPU improves data transfer efficiency
    """
    def __init__(self, env, crop=False):
        super(ProcessFrame84, self).__init__(env)
        self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84, 1))
        self._crop = crop

    def _step(self, action):
        obs, reward, done, info = self.env.step(action)
        return _process_frame84(obs, crop=self._crop), reward, done, info

    def _reset(self):
        return _process_frame84(self.env.reset(), crop=self._crop)


class RescaleFrameFloat(gym.Wrapper):
    """
    Rescale image frame from uint to a float between [0, 1]

    Note:
      Rescale on GPU instead of CPU improves data transfer efficiency
    """
    def __init__(self, env):
        super(RescaleFrameFloat, self).__init__(env)
        space = self.observation_space
        assert isinstance(space, spaces.Box), 'rescaling must be performed on spaces.Box'
        assert (space.low == 0).all() and (space.high == 255).all(), \
            'Should be 0 and 255:\n{}\n{}'.format(space.low, space.high)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=space.shape)
    
    @staticmethod
    def scale(obs):
        if obs.dtype != np.float32:
            obs = obs.astype(np.float32)
        return obs / 255.
    
    def _step(self, action):
        obs, reward, done, info = self.env.step(action)
        return RescaleFrameFloat.scale(obs), reward, done, info

    def _reset(self):
        return RescaleFrameFloat.scale(self.env.reset())


class ClippedRewardsWrapper(gym.Wrapper):
    def _step(self, action):
        obs, reward, done, info = self.env.step(action)
        return obs, np.sign(reward), done, info


# TODO: set all scale_float=False and use tf.cast instead across all envs
def wrap_deepmind(env, mode, scale_float=True, crop='auto', use_stack=True):
    """
    Typical: Pong, Breakout, SpaceInvaders, Seaquest, BeamRider, Enduro, Qbert
    
    mode: 'train' or 'test'
    
    crop: 
        True - downsample and crop
        False - direct downsampling
        'auto' - only crop Pong and Breakout
    """
    assert 'NoFrameskip' in env.spec.id
    is_train = (mode == 'train')

    if crop == 'auto':
        NO_CROP_GAMES = [] # not sure whether Qbert should be cropped or not
        crop = not any((game in env.spec.id) for game in NO_CROP_GAMES)
        print('wrap_deepmind: crop =', crop)
    if is_train:
        env = EpisodicLifeEnv(env)
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4, max=True)
    env = FireResetEnv(env)
    env = ProcessFrame84(env, crop=crop)
    if use_stack:
        env = StackFrameWrapper(env, buff=4)
    if is_train:
        env = ClippedRewardsWrapper(env)
    if scale_float:
        env = RescaleFrameFloat(env)
    return env
