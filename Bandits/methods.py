import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

## Consts
N_ITER = 1_000_000
N_ARMS = 10

LR = 0.01 ## Large value benefits greedy. Learns the greedy quickly and anchors.
WA = False ## Set to true if you want estimates to converge to a weighted average
EPS = 0.1 ## Percentage of choosing a random action in epsilon greedy.
C = 1 ## 0 == Greedy. 2 == emphasis on arms seldom tried.
Q_0 = 5 ## Probably the most interesting one. Set to value beyond range of random normal. Greedy has to explore those before deciding on the greedy action.

actuals = np.random.normal(size=N_ARMS)

def get_reward(action):
    return (actuals + np.random.normal())[action]

def greedy_pick(estimates, counts, t):
    return np.argmax(estimates)

def greedy_update(action, reward, estimates, counts, t):
    counts[action] = counts[action] + 1
    weight = (1 / counts[action]) if WA else LR
    estimates[action] = estimates[action] + weight * (reward - estimates[action])

def epsilon_greedy_pick(estimates, counts, t):
    x = np.random.random()
    return (x > EPS) * np.argmax(estimates) + (x <= EPS) * np.random.randint(0, N_ARMS)

def epsilon_greedy_update(action, reward, estimates, counts, t):
    counts[action] = counts[action] + 1
    weight = (1 / counts[action]) if WA else LR
    estimates[action] = estimates[action] + weight * (reward - estimates[action])

def ucb_pick(estimates, counts, t):
    return np.argmax(estimates + C * np.sqrt(np.log(t) / counts))

def ucb_update(action, reward, estimates, counts, t):
    counts[action] = counts[action] + 1
    weight = (1 / counts[action]) if WA else LR
    estimates[action] = estimates[action] + weight * (reward - estimates[action])

def gradient_pick(preferences, counts, t):
    p = np.exp(preferences) / np.exp(preferences).sum()
    return np.random.choice(np.arange(N_ARMS), p=p)

def gradient_update(action, reward, preferences, counts, t):
    p = np.exp(preferences) / np.exp(preferences).sum()
    preferences[action] = preferences[action] + LR * (reward - preferences[action]) * (1 - p[action])
    include = np.ones(N_ARMS)
    include[action] = 0
    preferences = preferences - LR * (reward - preferences) * p * include

def train_agent(pick_fn, update_fn):

    estimates = np.ones(shape=N_ARMS) * Q_0
    counts = np.zeros(shape=N_ARMS)
    data = np.zeros(shape=(N_ITER, 2))

    for t in range(1, N_ITER+1):
        
        action = pick_fn(estimates, counts, t)
        reward = get_reward(action)

        update_fn(action, reward, estimates, counts, t)        
        data[t-1] = [action, reward]

    return pd.DataFrame(data, columns = ['action', 'reward'])

if __name__ == '__main__':

    greedy_data = train_agent(greedy_pick, greedy_update)
    epsilon_greedy_data = train_agent(epsilon_greedy_pick, epsilon_greedy_update)
    ucb_data = train_agent(ucb_pick, ucb_update)
    gradient_data = train_agent(gradient_pick, gradient_update)

    print("Greedy")
    print(greedy_data.reward.sum())
    print(greedy_data.action.value_counts())
    print()
    print("E-Greedy")
    print(epsilon_greedy_data.reward.sum())
    print(epsilon_greedy_data.action.value_counts())
    print()
    print("UCB")
    print(ucb_data.reward.sum())
    print(ucb_data.action.value_counts())
    print()
    print("Gradient")
    print(gradient_data.reward.sum())
    print(gradient_data.action.value_counts())
    print()

    f, ax = plt.subplots(figsize=(14, 7))
    ax.plot(1+np.arange(N_ITER), greedy_data.reward.cumsum() / N_ITER, label="Greedy")
    ax.plot(1+np.arange(N_ITER), epsilon_greedy_data.reward.cumsum() / N_ITER, label="Eps-Greedy")
    ax.plot(1+np.arange(N_ITER), ucb_data.reward.cumsum() / N_ITER, label="UCB")
    ax.plot(1+np.arange(N_ITER), gradient_data.reward.cumsum() / N_ITER, label="Gradient")
    ax.legend()
    ax.set_xscale("log")
    ax.set_xlabel("t")
    ax.set_ylabel("Average Reward")
    plt.show()

    f, ax = plt.subplots(figsize=(14, 7))
    best_action = np.argmax(actuals)
    ax.plot(1+np.arange(N_ITER), (greedy_data.action == best_action).cumsum() / N_ITER, label="Greedy")
    ax.plot(1+np.arange(N_ITER), (epsilon_greedy_data.action == best_action).cumsum() / N_ITER, label="Eps-Greedy")
    ax.plot(1+np.arange(N_ITER), (ucb_data.action == best_action).cumsum() / N_ITER, label="UCB")
    ax.plot(1+np.arange(N_ITER), (gradient_data.action == best_action).cumsum() / N_ITER, label="Gradient")
    ax.legend()
    ax.set_xlabel("t")
    ax.set_xscale("log")
    ax.set_ylabel("% of Time Selecting Optimal Action")
    plt.show()