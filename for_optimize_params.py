# A "n-step SARSA" implementation
from collections import deque
from math import gamma, exp
import gym
import numpy as np
from functools import reduce
import random

from util_plot import plot_result
from util_plot import plot_result, plot_multiple_results
from util_experiments import test_greedy_Q_policy, repeated_exec_steps

# Softmax policy
def softmax(x):
    temp_max = max(x)
    x = [i - temp_max for i in x]
    x = [exp(i) for i in x]
    temp_sum = sum(x)
    x = [i / temp_sum for i in x]
    return x
# escolhe uma ação da Q-table usando uma estratégia softmax
def softmax_policy(Q, state):
    probs = softmax(Q[state])
    return np.random.choice(len(probs), p=probs)
def importance_sampling(Q, state, action, num_actions, epsilon):
    TARGET = softmax(Q[state])
    return TARGET[action]/(epsilon / num_actions)
def epsilon_greedy_probs(Q, state, num_actions, epsilon):
    # probabilidade que todas as ações têm de ser escolhidas nas decisões exploratórias (não-gulosas)
    non_greedy_action_probability = epsilon / num_actions
    # conta quantas ações estão empatadas com o valor máximo de Q neste estado
    q_max = np.max(Q[state, :])
    greedy_actions = 1
    for i in range(num_actions):
        if Q[state][i] == q_max:
            greedy_actions += 1
    # probabilidade de cada ação empatada com Q máximo:
    # probabilidade de ser escolhida de forma gulosa (greedy) + probabilidade de ser escolhida de forma exploratória
    greedy_action_probability = (
        (1 - epsilon) / greedy_actions) + non_greedy_action_probability
    # prepara a lista de probabilidades: cada índice tem a probabilidade da ação daquele índice
    probs = []
    for i in range(num_actions):
        if Q[state][i] == q_max:
            probs.append(greedy_action_probability)
        else:
            probs.append(non_greedy_action_probability)
    return probs
# Esta é a política. Neste caso, escolhe uma ação com base nos valores
# da tabela Q, usando uma estratégia epsilon-greedy.
def choose_action(Q, state, num_actions, epsilon):
    if np.random.random() < epsilon:
        return np.random.randint(0, num_actions)
    else:
        # alt. para aleatorizar empates: np.random.choice(np.where(b == bmax)[0])
        return np.argmax(Q[state])
# Algoritmo "n-step SARSA", online learning
# Atenção: os espaços de estados e de ações precisam ser discretos, dados por valores inteiros
def run_nstep_sarsa_offpolicy(env, episodes, nstep=1, lr=0.1, gamma=0.95, epsilon=0.1, render=False):
    assert isinstance(env.observation_space, gym.spaces.Discrete)
    assert isinstance(env.action_space, gym.spaces.Discrete)
    num_actions = env.action_space.n
    # inicializa a tabela Q com valores aleatórios de -1.0 a 0.0
    # usar o estado como índice das linhas e a ação como índice das colunas
    # Q = np.random.uniform(low=-1.0, high=0.0,
    #                     size=(env.observation_space.n, num_actions))
    Q = [[random.uniform(0, 1) for _ in range(num_actions)]
         for _ in range(env.observation_space.n)]
    gamma_array = [gamma**i for i in range(0, nstep)]
    gamma_power_nstep = gamma**nstep
    # para cada episódio, guarda sua soma de recompensas (retorno não-discontado)
    sum_rewards_per_ep = []
    # loop principal
    for i in range(episodes):
        done = False
        sum_rewards, reward = 0, 0
        next_state = env.reset()
        # escolhe a próxima ação
        next_action = choose_action(Q, next_state, num_actions, epsilon)
        # históricos de: estados, ações e recompensas
        hs = deque(maxlen=nstep)
        ha = deque(maxlen=nstep)
        hr = deque(maxlen=nstep)
        P = []
        # executa 1 episódio completo, fazendo atualizações na Q-table
        while not done:
            # exibe/renderiza os passos no ambiente, durante 1 episódio a cada mil e também nos últimos 5 episódios
            if render and (i >= (episodes - 5) or (i+1) % 1000 == 0):
                env.render()
            # preparação para avançar mais um passo
            # lembrar que a ação a ser realizada já está escolhida
            state = next_state
            action = next_action
            # realiza a ação
            next_state, reward, done, _ = env.step(action)
            sum_rewards += reward
            hs.append(state)
            ha.append(action)
            hr.append(reward)
            P.append(importance_sampling(Q, next_state,
                                         next_action, num_actions, epsilon))
            # se o histórico estiver completo,
            # vai fazer uma atualização no valor Q do estado mais antigo
            if len(hs) == nstep:
                if done:
                    # para estados terminais
                    V_next_state = 0
                else:
                    # escolhe (antecipadamente) a ação do próximo estado
                    next_action = choose_action(
                        Q, next_state, num_actions, epsilon)
                    # para estados não-terminais -- valor máximo (melhor ação)
                    V_next_state = Q[next_state][next_action]
                # delta = (estimativa usando a nova recompensa) - estimativa antiga
                delta = (sum(i*y for i, y in zip(gamma_array, hr)) +
                         gamma_power_nstep * V_next_state) - Q[hs[0]][ha[0]]
                # atualiza a Q-table para o par (estado,ação) de n passos atrás
                Q[hs[0]][ha[0]] += lr * reduce(lambda x, y: x*y, P) * delta
            # fim do laço por episódio
        # ao fim do episódio, atualiza o Q dos estados que restaram no histórico
        # pode ser inferior ao "nstep", em episódios muito curtos
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
        # a cada 100 episódios, imprime informação sobre o progresso
        # if (i+1) % 100 == 0:
        #    avg_reward = np.mean(sum_rewards_per_ep[-100:])
        #    print(f"Episode {i+1} Average Reward (last 100): {avg_reward:.3f}")
    return sum_rewards_per_ep, Q
def calculate_G(gamma_array, hr, V_next_state, P, nstep):
    G = 0
    for step in range(nstep):
        G += P[step] * (hr[step] + gamma_array[step] * G) + \
            (1 - P[step]) * V_next_state
    return G
# Algoritmo "n-step SARSA", online learning
# Atenção: os espaços de estados e de ações precisam ser discretos, dados por valores inteiros
def run_nstep_sarsa_cv(env, steps, nstep=1, lr=0.1, gamma=0.95, epsilon=0.1, render=False):
    assert isinstance(env.observation_space, gym.spaces.Discrete)
    assert isinstance(env.action_space, gym.spaces.Discrete)
    num_actions = env.action_space.n
    # inicializa a tabela Q com valores aleatórios de -1.0 a 0.0
    # usar o estado como índice das linhas e a ação como índice das colunas
    Q = [[random.uniform(0, 1) for _ in range(num_actions)]
         for _ in range(env.observation_space.n)]
    gamma_array = [gamma**i for i in range(0, nstep)]
    # para cada episódio, guarda sua soma de recompensas (retorno não-discontado)
    sum_rewards_per_ep = []
    done = False
    sum_rewards, reward = 0, 0
    next_state = env.reset()
    # escolhe a próxima ação
    next_action = choose_action(Q, next_state, num_actions, epsilon)
    # históricos de: estados, ações e recompensas
    hs = deque(maxlen=nstep)
    ha = deque(maxlen=nstep)
    hr = deque(maxlen=nstep)
    P = []
    FINISH_STEP = 0
    # executa 1 episódio completo, fazendo atualizações na Q-table
    for step in range(steps):
        FINISH_STEP += step
        # exibe/renderiza os passos no ambiente, durante 1 episódio a cada mil e também nos últimos 5 episódios
        # if render and (i >= (episodes - 5) or (i+1) % 1000 == 0):
        #    env.render()
        # preparação para avançar mais um passo
        # lembrar que a ação a ser realizada já está escolhida
        state = next_state
        action = next_action
        # realiza a ação
        next_state, reward, done, _ = env.step(action)
        sum_rewards += reward
        hs.append(state)
        ha.append(action)
        hr.append(reward)
        P.append(importance_sampling(Q, next_state,
                                     next_action, num_actions, epsilon))
        # se o histórico estiver completo,
        # vai fazer uma atualização no valor Q do estado mais antigo
        if len(hs) == nstep:
            if done:
                #FINISH_STEP = step
                # para estados terminais
                V_next_state = 0
            else:
                # escolhe (antecipadamente) a ação do próximo estado
                next_action = choose_action(
                    Q, next_state, num_actions, epsilon)
                # para estados não-terminais -- valor máximo (melhor ação)
                V_next_state = Q[next_state][next_action]
            # delta = (estimativa usando a nova recompensa) - estimativa antiga
            # G = sum(gamma_array*hr) + gamma_power_nstep * V_next_state
            delta = calculate_G(
                gamma_array, hr, V_next_state, P, nstep) - Q[hs[0]][ha[0]]
            # atualiza a Q-table para o par (estado,ação) de n passos atrás
            Q[hs[0]][ha[0]] += lr * delta
        # fim do laço por episódio
    # ao fim do episódio, atualiza o Q dos estados que restaram no histórico
    # pode ser inferior ao "nstep", em episódios muito curtos
    laststeps = len(hs)
    for j in range(laststeps-1, 0, -1):
        hs.popleft()
        ha.popleft()
        hr.popleft()
        P.pop(len(P)-1)
        delta = calculate_G(gamma_array, hr, 0, P, j) - Q[hs[0]][ha[0]]
        Q[hs[0]][ha[0]] += lr * delta
    sum_rewards_per_ep.append((FINISH_STEP-1, sum_rewards))
    # a cada 100 episódios, imprime informação sobre o progresso
    # if (i+1) % 100 == 0:
    #    avg_reward = np.mean(sum_rewards_per_ep[-100:])
    #    print(f"Episode {i+1} Average Reward (last 100): {avg_reward:.3f}")
    return sum_rewards_per_ep, Q,
