# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # OpenAI Gym Tutorial

# %%
import gym
import numpy as np
import matplotlib.pyplot as plt
from FrozenLake import FrozenLakeEnv
from collections import Counter


MAP_SIZE="8x8"
if MAP_SIZE == "4x4":
    goal = 15
else:
    goal = 63
# %%
# env = gym.make('FrozenLake-v1', map_name="4x4")
env = FrozenLakeEnv(map_name="8x8", is_slippery=True)
env.reset()
env.render()


# %%
rewards = []
actions = {
    0: "LEFT",
    1: "DOWN",
    2: "RIGHT",
    3: "UP"
}

Sn = len(env.P)
grid_size = int(np.sqrt(Sn))

# %%
def value_iteration(env, gamma=1, theta=1e-8):
    r"""Policy evaluation function. Loop until state rewards are stable.
    
    Returns: 
        V (np.array): expected state value given an infinite horizon.
        policy (np.array): best action for each state.

    Args:
        env (gym.env): gym environment.
        gamma (float): future reward discount rate.
        theta (float): stopping criterion.
    """
    # Initialize state-value array
    S_n = env.observation_space.n
    V = np.zeros(S_n)
    policy = np.ones(S_n) * -1
    delta = np.zeros(S_n)
    i = 0
    while True:
        i += 1
        # Loop through states
        for s in env.P:
            Vs = np.zeros(len(env.P[s]))
            # Loop through available actions in each state
            for a in env.P[s]:
                # Loop though transition probabilities, next state, and rewards
                for prob, next_state, reward, done in env.P[s][a]:
                    Vs[a] += prob * (reward + gamma * V[next_state])
            delta[s] = np.abs(V[s] - Vs.max())
            V[s] = Vs.max()
            policy[s] = Vs.argmax()
        if np.all(delta < theta):
            print("Value iteration complete after {} steps".format(i))
            break
    return V, policy


# %%
V, policy = value_iteration(env, gamma=.9)

print(np.array2string(V.reshape(grid_size,grid_size), precision=3))
print(np.array2string(np.array([actions[a] for a in policy]).reshape(grid_size,grid_size)))


# %%
total_reward = 0
episodes = 1000
end_state = []
episode_failures = []
for i_episode in range(episodes):
    obs = env.reset()
    action_list = []
    state_list = [obs]
    for t in range(100):
        # env.render()
        # print(obs)
        action = policy[obs]
        action_list.append(action)
        obs, reward, done, info = env.step(action)  # take an action based on the optimal policy
        state_list.append(obs)
        if done:
            # total_reward += reward
            end_state.append(obs)
            if obs == goal:
                total_reward += 1
                # print ("Made the goal!")
            else:
                # Add the action sequence that produced the bad outcome
                episode_failures.append((state_list, action_list))
                # print("Fell in the hole at ({},{})".format(obs // 4, obs%4))
            # print("Episode {} finished after {} timeteps. Reward {}.".format(
                # i_episode+1, t+1, reward))
            break
env.close()
print(f"Reward percentage {total_reward/episodes}")

histogram = Counter(end_state)
locs = [f"({idx // grid_size}, {idx % grid_size})" for idx in histogram.keys()]
plt.bar(locs, histogram.values())
plt.title("Terminal State locations")
plt.show()
