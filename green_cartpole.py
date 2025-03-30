import random
import numpy as np 

from gymnasium import spaces
from gymnasium.envs.classic_control import CartPoleEnv as CartPoleEnvOriginal
from gymnasium.envs.registration import EnvSpec
from typing import Optional

class logger ():

    def __init__ ():
        pass
    def warn(message):

        print(message)

def neuralnet_dummy(s,a):
    '''
    dummy neural network that just multiplies every element in s by
    a random number between 0 and 1, ignoring a
    | Num | Observation           | Min                 | Max               |
    |-----|-----------------------|---------------------|-------------------|
    | 0   | Cart Position         | -4.8                | 4.8               |
    | 1   | Cart Velocity         | -Inf                | Inf               |
    | 2   | Pole Angle            | ~ -0.418 rad (-24°) | ~ 0.418 rad (24°) |
    | 3   | Pole Angular Velocity | -Inf                | Inf               |

    '''
    #state = [sprime * np.random.normal(1,0.1) for sprime in s]
    #state = [sprime * 1.1 for sprime in s]
     
    #state = tuple(state)
    
    return s
    #return s
    


class CartPoleEnv(CartPoleEnvOriginal):
    """Same as CartPoleEnv but with spec."""

    def __init__(self, render_mode):
        super().__init__(render_mode=render_mode)
        self._spec = EnvSpec(id="CartPoleEnv")

    @property
    def spec(self):
        """Return the spec of the environment.

        Note: spec.maximum_nr_steps is needed by rllib to truncate the episode.
        """
        return self._spec


class GreenCartPoleEnv(CartPoleEnv):  # pylint: disable=invalid-name   
    """A partial observable environment for the CartPole problem, where angular and posiional
    velocity are not observable. The state of the environment is the position of the cart and the
    angle.
    """

    def step(self, action):
        assert self.action_space.contains(
            action
        ), f"{action!r} ({type(action)}) invalid"
        assert self.state is not None, "Call reset before using step method."
        
        
        ####### replace from here ###########
        """
        force = self.force_mag if action == 1 else -self.force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (
            force + self.polemass_length * theta_dot**2 * sintheta
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        self.state = (x, x_dot, theta, theta_dot)
        """
        ###### until here #################

        # implement randomizer in the physics upper and lower bounds: 

        self.state = neuralnet_dummy(self.state, action)

        x, x_dot, theta, theta_dot = self.state


        # stays the same : 
        terminated = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )
        # stays the same 
        if not terminated:
            reward = 1.0
        elif self.steps_beyond_terminated is None:
            # Pole just fell!
            self.steps_beyond_terminated = 0
            reward = 1.0
        else:
            if self.steps_beyond_terminated == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned terminated = True. You "
                    "should always call 'reset()' once you receive 'terminated = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_terminated += 1
            reward = 0.0

        if self.render_mode == "human":
            self.render()
        return np.array(self.state, dtype=np.float32), reward, terminated, False, {}
      

if __name__ == "__main__":
    
    print(CartPoleEnv(render_mode="human"))