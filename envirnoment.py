from green_cartpole import GreenCartPoleEnv
import random

from gymnasium.wrappers import TimeLimit


envirnoment = TimeLimit(GreenCartPoleEnv(render_mode='human'),max_episode_steps=200)
envirnoment.reset()

for count in range(10000):
    print(count)
    step = random.randint(0,1)
    state, reward , terminated , truncated , info = envirnoment.step(step)
    if truncated :
        print(count)
        print( 'truncated ')
        break
    elif terminated :
        print(count)
        print ('terminated ')
        break

print('end')