from collections import deque
from math import exp
import gym
import numpy as np
from functools import reduce
import random
from util_experiments import test_greedy_Q_policy, test_greedy_Q_policy_steps
from wrappers import DiscreteObservationWrapper

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


def run_nstep_sarsa_offPolicy(env, episodes, nstep=1, lr=0.1, gamma=0.95, epsilon=0.1, render=False):
    assert isinstance(env.observation_space, gym.spaces.Discrete)
    assert isinstance(env.action_space, gym.spaces.Discrete)
    num_actions = env.action_space.n
    Q = [[random.uniform(0, 1) for _ in range(num_actions)]
         for _ in range(env.observation_space.n)]
    gamma_array = [gamma**i for i in range(0, nstep)]
    gamma_power_nstep = gamma**nstep
    sum_rewards_per_ep = []

    for i in range(episodes):
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

    mean_return, episode_return = test_greedy_Q_policy(env, Q, episodes, False)
    return episode_return, mean_return


def run_nstep_sarsa_offPolicy_steps(env, steps, nstep=1, lr=0.1, gamma=0.95, epsilon=0.1, render=False):
    assert isinstance(env.observation_space, gym.spaces.Discrete)
    assert isinstance(env.action_space, gym.spaces.Discrete)
    num_actions = env.action_space.n
    Q = [[random.uniform(0, 1) for _ in range(num_actions)]
         for _ in range(env.observation_space.n)]
    gamma_array = [gamma**i for i in range(0, nstep)]
    gamma_power_nstep = gamma**nstep
    sum_rewards_per_ep = []
    done = False
    sum_rewards, reward = 0, 0
    next_state = env.reset()
    next_action = choose_action(Q, next_state, num_actions, epsilon)
    hs = deque(maxlen=nstep)
    ha = deque(maxlen=nstep)
    hr = deque(maxlen=nstep)
    P = []

    for step in range(steps):
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
                next_state = env.reset()
                sum_rewards_per_ep.append((step, sum_rewards))
                sum_rewards = 0
    
            else:
                next_action = choose_action(
                    Q, next_state, num_actions, epsilon)
                V_next_state = Q[next_state][next_action]

            delta = (sum(i*y for i, y in zip(gamma_array, hr)) +
                     gamma_power_nstep * V_next_state) - Q[hs[0]][ha[0]]
            Q[hs[0]][ha[0]] += lr * reduce(lambda x, y: x*y, P) * delta

            if not done:
                sum_rewards_per_ep.append((step, sum_rewards))
        if done:
            next_state = env.reset()
            sum_rewards_per_ep.append((step, sum_rewards))
            sum_rewards = 0
            laststeps = len(hs)
            for j in range(laststeps-1, 0, -1):
                hs.popleft()
                ha.popleft()
                hr.popleft()
                P.pop(len(P)-1)
                delta = (
                    sum(i*y for i, y in zip(gamma_array[:j], hr)) + 0) - Q[hs[0]][ha[0]]
                Q[hs[0]][ha[0]] += lr * reduce(lambda x, y: x*y, P) * delta


    all_return = test_greedy_Q_policy_steps(env, Q, steps, False)
        
    return all_return


def calculate_G(gamma_array, hr, V_next_state, P, nstep):
    G = 0
    for step in range(nstep):
        G += P[step] * (hr[step] + gamma_array[step] * G) + \
            (1 - P[step]) * V_next_state
    return G


def run_nstep_sarsa_offPolicy_control_variate(env, episodes, nstep=1, lr=0.1, gamma=0.95, epsilon=0.1, render=False):
    assert isinstance(env.observation_space, gym.spaces.Discrete)
    assert isinstance(env.action_space, gym.spaces.Discrete)
    num_actions = env.action_space.n
    Q = [[random.uniform(0, 1) for _ in range(num_actions)]
         for _ in range(env.observation_space.n)]
    gamma_array = [gamma**i for i in range(0, nstep)]
    sum_rewards_per_ep = []

    for i in range(episodes):
        done = False
        sum_rewards, reward = 0, 0

        next_state = env.reset()
        next_action = choose_action(Q, next_state, num_actions, epsilon)
        hs = deque(maxlen=nstep)
        ha = deque(maxlen=nstep)
        hr = deque(maxlen=nstep)
        P = []
        FINISH_STEP = 0
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

        sum_rewards_per_ep.append((FINISH_STEP-1, sum_rewards))

    mean_return, episode_return = test_greedy_Q_policy(env, Q, episodes, False)
    return episode_return, mean_return


def run_nstep_sarsa_offPolicy_control_variate_steps(env, steps, nstep=1, lr=0.1, gamma=0.95, epsilon=0.1, render=False):
    assert isinstance(env.observation_space, gym.spaces.Discrete)
    assert isinstance(env.action_space, gym.spaces.Discrete)
    episodes = 0
    num_actions = env.action_space.n
    Q = [[random.uniform(0, 1) for _ in range(num_actions)]
         for _ in range(env.observation_space.n)]
    gamma_array = [gamma**i for i in range(0, nstep)]
    sum_rewards_per_ep = []
    done = False
    sum_rewards, reward = 0, 0
    next_state = env.reset()
    next_action = choose_action(Q, next_state, num_actions, epsilon)
    hs = deque(maxlen=nstep)
    ha = deque(maxlen=nstep)
    hr = deque(maxlen=nstep)
    P = []

    for step in range(steps):
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
                next_state = env.reset()
                episodes += 1
                sum_rewards = 0
                V_next_state = 0
                laststeps = len(hs)
                for j in range(laststeps-1, 0, -1):
                    hs.popleft()
                    ha.popleft()
                    hr.popleft()
                    P.pop(len(P)-1)
                    delta = calculate_G(gamma_array, hr, 0,
                                        P, j) - Q[hs[0]][ha[0]]
                    Q[hs[0]][ha[0]] += lr * delta

            else:
                next_action = choose_action(
                    Q, next_state, num_actions, epsilon)
                V_next_state = Q[next_state][next_action]

            delta = calculate_G(
                gamma_array, hr, V_next_state, P, nstep) - Q[hs[0]][ha[0]]

            Q[hs[0]][ha[0]] += lr * delta

            if not done:
                sum_rewards_per_ep.append((step, sum_rewards))

    mean_return, episode_return = test_greedy_Q_policy_steps(env, Q, steps, False)
    return episode_return, mean_return

if __name__ == "__main__":
    ENV_NAME = "CartPole-v1"
    EPISODES = 10000
    LR = 0.4332311032873412
    GAMMA = 0.8488928377337801
    EPSILON = 1
    NSTEPS = 20
    B1 = 1
    B2= 69
    B3= 13
    B4 = 68

    env = gym.make(ENV_NAME)
    env = DiscreteObservationWrapper(env, [B1,B2,B3,B4])

    rewards, Qtable = run_nstep_sarsa_offPolicy_steps(
       env, 50, NSTEPS, LR, GAMMA, EPSILON, render=False)
    print("Últimos resultados: media =", np.mean(
       rewards[-20:]), ", desvio padrao =", np.std(rewards[-20:]))