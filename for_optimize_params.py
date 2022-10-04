from collections import deque
from math import gamma, exp
import gym
import numpy as np
from functools import reduce
import random


def softmax(x):
    temp_max = max(x)
    x = [i - temp_max for i in x]
    x = [exp(i) for i in x]
    temp_sum = sum(x)
    x = [i / temp_sum for i in x]
    return x

def importance_sampling(Q, state, action, num_actions, epsilon):
    TARGET = softmax(Q[state])
    return TARGET[action]/(epsilon / num_actions)
    

def choose_action(Q, state, num_actions, epsilon):
    if np.random.random() < epsilon:
        return np.random.randint(0, num_actions)
    else:
        return np.argmax(Q[state])


def calculate_G(gamma_array, hr, V_next_state, P, nstep):
    G = 0
    for step in range(nstep):
        G += P[step] * (hr[step] + gamma_array[step] * G) + \
            (1 - P[step]) * V_next_state
    return G
        

def run_nstep_sarsa_offpolicy(env, episodes, nstep=1, lr=0.1, gamma=0.95, epsilon=0.1, render=False):
    assert isinstance(env.observation_space, gym.spaces.Discrete)
    assert isinstance(env.action_space, gym.spaces.Discrete)
    num_actions = env.action_space.n
    Q = [[random.uniform(0, 1) for _ in range(num_actions)]
         for _ in range(env.observation_space.n)]
    gamma_array = [gamma**i for i in range(0, nstep)]
    gamma_power_nstep = gamma**nstep
    
    sum_rewards_per_ep = []

    for _ in range(episodes):
        done = False
        sum_rewards, reward = 0, 0
        next_state = env.reset()
        next_action = choose_action(Q, next_state, num_actions, epsilon)
        hs = deque(maxlen=nstep)
        ha = deque(maxlen=nstep)
        hr = deque(maxlen=nstep)
        P = []

        while not done:
            state = next_state
            action = next_action
            next_state, reward, done, _ = env.step(action)
            sum_rewards += reward
            hs.append(state)
            ha.append(action)
            hr.append(reward)
            P.append(importance_sampling(Q, next_state,
                                         next_action, num_actions, epsilon))
            if len(hs) == nstep:
                if done:
                    V_next_state = 0

                else:
                    next_action = choose_action(Q, next_state, num_actions, epsilon)
                    V_next_state = Q[next_state][next_action]

                delta = (sum(i*y for i, y in zip(gamma_array, hr)) +
                         gamma_power_nstep * V_next_state) - Q[hs[0]][ha[0]]
                Q[hs[0]][ha[0]] += lr * reduce(lambda x, y: x*y, P) * delta
                
        laststeps = len(hs)
        for j in range(laststeps-1, 0, -1):
            hs.popleft()
            ha.popleft()
            hr.popleft()
            P.pop(len(P)-1)
            delta = (
                sum(i*y for i, y in zip(gamma_array[:j], hr)) + 0) - Q[hs[0]][ha[0]]
            Q[hs[0]][ha[0]] += lr * reduce(lambda x, y: x*y, P) * delta
        sum_rewards_per_ep.append(sum_rewards)
    return sum_rewards_per_ep, Q


def run_nstep_sarsa_cv(env, episodes, nstep=1, lr=0.1, gamma=0.95, epsilon=0.1, render=False):
    assert isinstance(env.observation_space, gym.spaces.Discrete)
    assert isinstance(env.action_space, gym.spaces.Discrete)
    num_actions = env.action_space.n
    Q = [[random.uniform(0, 1) for _ in range(num_actions)]
         for _ in range(env.observation_space.n)]
    gamma_array = [gamma**i for i in range(0, nstep)]
    sum_rewards_per_ep = []

    for _ in range(episodes):
        done = False
        sum_rewards, reward = 0, 0
        next_state = env.reset()
        next_action = choose_action(Q, next_state, num_actions, epsilon)
        hs = deque(maxlen=nstep)
        ha = deque(maxlen=nstep)
        hr = deque(maxlen=nstep)
        P = []

        while not done:
            state = next_state
            action = next_action
            next_state, reward, done, _ = env.step(action)
            sum_rewards += reward
            hs.append(state)
            ha.append(action)
            hr.append(reward)
            P.append(importance_sampling(Q, next_state,
                                        next_action, num_actions, epsilon))
            if len(hs) == nstep:
                if done:
                    V_next_state = 0
                else:
                    next_action = choose_action(
                        Q, next_state, num_actions, epsilon)
                    V_next_state = Q[next_state][next_action]
                delta = calculate_G(
                    gamma_array, hr, V_next_state, P, nstep) - Q[hs[0]][ha[0]]
                Q[hs[0]][ha[0]] += lr * delta

        laststeps = len(hs)

        for j in range(laststeps-1, 0, -1):
            hs.popleft()
            ha.popleft()
            hr.popleft()
            P.pop(len(P)-1)
            delta = calculate_G(gamma_array, hr, 0, P, j) - Q[hs[0]][ha[0]]
            Q[hs[0]][ha[0]] += lr * delta

        sum_rewards_per_ep.append(sum_rewards)

    return sum_rewards_per_ep, Q,
