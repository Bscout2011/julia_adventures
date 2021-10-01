# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Gambler's Problem
# 
# Consider a gamble on a series of coin flips. The goal is to get $100 from a set of capital $s \in \{1, 2, \ldots, 99 \}$ by wagering an action of $a \in \{0, 1, \ldots, \min(s, 100-s) \}$. The reward is zero for all bets until the goal of $100 is reached which gives +1. How much should the gambler bet in each round with their captial?
# 
# Given probability of positively flipping a coin $p_h$, the solution can be computed with *value iteration*
# 
# $$
# V(s) = \max_a \sum_{s'} p(s', r \; | \; s, a) [r + \gamma V(s')].
# $$
# 
# In English, this equation states the optimal action $a$ in the current state $s$ is the action which lands in the next state $s'$ with reward $r$ along with the discounted reward $\gamma V(s')$. Since each action has several outcomes $p(s' \; | s, a)$, the sum term computes the expected reward from all possible outcomes. Also the equation recursively calls itself $\gamma V(s')$. It's also non-linear because we're chosing the action with maximum expected reward $\max_a$.

# %%
import numpy as np
import matplotlib.pyplot as plt


# %%
capital = list(range(100+1))
def stakes(s):
    return list(range(min(s, 100 - s)+1))


# %%
ph = 0.4
gamma = .99

Sn = len(capital)
V = np.zeros(Sn)
policy = np.zeros(Sn)
iter = 0
theta = 0

# save V at iterations 1, 2, 3, 32
Vi = []

while True:
    delta = np.zeros(Sn)
    iter += 1
    for i, s in enumerate(capital):
        if s == 100:
            continue
        actions = stakes(s)
        Vs = np.zeros(len(actions))
        for j, a in enumerate(actions):
            # for each action, the outcomes are only positive if s+a = 100
            r = 0
            # if a + s > 100:
            #     continue  # out of domain condition
            if (a + s) == 100:
                r = 1
                # print(f"Winning combo: {a}+{s}")
            
            Vs[j] = ph * (r + gamma * V[a+s])
            # Add expected reward from tail outcome
            # if s-a > 0:
            Vs[j] += (1 - ph) * (gamma * V[s - a])

        delta[i] = np.abs(V[i] - Vs.max())
        V[i] = Vs.max()
        policy[i] = Vs.argmax()

    if iter in (1, 2, 3, 32):
        Vi.append(V.copy())    
    
    if np.all(delta <= theta):
        print("Value iteration complete after {} steps".format(iter))
        break


# %%
fig, axs = plt.subplots(2, 1)
axs[0].plot(capital, V)
for val in Vi:
    axs[0].plot(capital, val)
axs[1].plot(capital, policy)
axs[0].set_ylabel("Expected Value")
axs[1].set_ylabel("Optimal Stake")
axs[1].set_xlabel("Capital")
axs[0].set_title("Value Iteration solution for the Gambler's Problem")
plt.show()


