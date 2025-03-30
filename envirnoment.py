import gym
from gym import Env
from gym.spaces import Discrete, Box
import numpy as np

class CustomCartPoleEnv(Env):
    def __init__(self):
        super(CustomCartPoleEnv, self).__init__()
        # Create the original CartPole environment
        #self.env = gym.make('CartPole-v1')

        # Define action and observation space based on the wrapped env
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def step(self, action):
        # Use the wrapped env's step
        obs, reward, done, info = self.env.step(action)
        
        # You can modify the reward or observation here if needed
        return obs, reward, done, info

    def reset(self):
        return self.env.reset()

    def render(self, mode="human"):
        self.env.render()

    def close(self):
        self.env.close()
