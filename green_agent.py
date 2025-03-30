"""Your first learning agent."""
import gymnasium as gym
from matplotlib import pyplot as plt
from plot_util import visualize_env
from ray.rllib.algorithms.dqn import DQNConfig

from ray.tune.registry import register_env
from green_cartpole import GreenCartPoleEnv
from gymnasium.wrappers import TimeLimit

# register our costum envirnoment 
def env_creator(config):
    #wrapp it in timelimit 
    return TimeLimit(GreenCartPoleEnv(config)  , max_episode_steps=200) # Return a gymnasium.Env instance.
register_env("green_env", env_creator)


# 1 - Build an agent
# 1.1 - Get the default config of xxxConfig()

config = DQNConfig()
# 1.2 - Examine the config by converting it to a dict via .to_dict() method
config_as_dict = config.to_dict()

# 1.3 - Modify the config if needed, e.g. change the "num_gpus" to 0, or change the learning_rate
# (lr)
# 1.4 - Introduce the environment to the agent's config
config.environment(env="green_env").framework(
    framework="tf2", eager_tracing=True
).rollouts(num_rollout_workers=4, num_envs_per_worker=2).evaluation(
    evaluation_config={"explore": False},
    evaluation_duration=10,
    evaluation_interval=1,
    evaluation_duration_unit="episodes",
)

# 1.5 - Build the agent from the config with .build()

agent = config.build()
# 2 - Train the agent for one training round with .train and get the reports
# reports = agent.train()
# print(reports)

# 3 - Run a loop for nr_trainings = 50 times
nr_trainings = 20  # pylint: disable=invalid-name
mean_rewards = []
for _ in range(nr_trainings):
    agent.train()
    print("total reward:", agent.evaluate()["evaluation"]["episode_reward_mean"])

# plot the mean rewards
plt.plot(mean_rewards)
plt.xlabel("Training rounds")
plt.ylabel("Mean reward")
plt.title("Mean reward vs. training rounds")
# save the plot
plt.savefig("mean_reward_vs_training_rounds_2.png")
plt.close()


# 4 - Visualize the trained agent; This is similar to running the random_agent,
# except that this time we have a trained agent
# 4.1 - Create an environment similar to the training env.
env = gym.make("CartPole-v1", render_mode="rgb_array")
s, _ = env.reset()
done = False  # pylint: disable=invalid-name
cumulative_reward = 0  # pylint: disable=invalid-name

while not done:
    # 4.2. Let the agent choose an action;
    a = agent.compute_single_action(observation=s, explore=False)
    # 4.3. and pass it to the environment
    s, r, terminated, truncated, info = env.step(action=a)

    # 4.4. How much reward did you get for that action? Keep the score!
    cumulative_reward += r
    done = terminated or truncated
    # 4.5. Repeat the 4.{2,3, 4} until the end of the episode
    # visualize the agent
    visualize_env(env=env, pause_sec=0.1)
    # continue with the next step without closing the plot

# 4.6. How much total reward you got? What does it mean to have large/small reward?
print("Total reward:", cumulative_reward)

print("Good-bye.")
