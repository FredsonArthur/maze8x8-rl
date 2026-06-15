import numpy as np
from collections import defaultdict

class QLearningAgent:
    """Agente que usa algoritmo Q-Learning"""
    
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99, epsilon=0.1):
        """
        Parâmetros:
        - n_states: número de estados (8x8 = 64)
        - n_actions: número de ações (4)
        - alpha: taxa de aprendizado
        - gamma: fator de desconto
        - epsilon: taxa de exploração (ε-greedy)
        """
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        
        # Tabela Q: dicionário que mapeia (estado, ação) -> valor
        self.Q = defaultdict(float)
    
    def get_action(self, state, training=True):
        """Escolhe ação usando política ε-greedy"""
        if training and np.random.random() < self.epsilon:
            # Exploração: escolhe ação aleatória
            return np.random.randint(self.n_actions)
        else:
            # Exploração: escolhe melhor ação baseada na Q-table
            return self.get_best_action(state)
    
    def get_best_action(self, state):
        """Retorna a melhor ação para um estado (greedy policy)"""
        values = [self.Q[(state, a)] for a in range(self.n_actions)]
        return np.argmax(values)
    
    def update(self, state, action, reward, next_state, done):
        """Atualiza a Q-table usando Q-Learning"""
        if done:
            target = reward
        else:
            # Q-Learning: usa max_a Q(next_state, a)
            next_values = [self.Q[(next_state, a)] for a in range(self.n_actions)]
            target = reward + self.gamma * max(next_values)
        
        # Atualização Q-learning
        self.Q[(state, action)] += self.alpha * (target - self.Q[(state, action)])


class SarsaAgent:
    """Agente que usa algoritmo SARSA (State-Action-Reward-State-Action)"""
    
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99, epsilon=0.1):
        """
        Parâmetros:
        - n_states: número de estados (8x8 = 64)
        - n_actions: número de ações (4)
        - alpha: taxa de aprendizado
        - gamma: fator de desconto
        - epsilon: taxa de exploração (ε-greedy)
        """
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        
        # Tabela Q: dicionário que mapeia (estado, ação) -> valor
        self.Q = defaultdict(float)
    
    def get_action(self, state, training=True):
        """Escolhe ação usando política ε-greedy"""
        if training and np.random.random() < self.epsilon:
            # Exploração: escolhe ação aleatória
            return np.random.randint(self.n_actions)
        else:
            # Exploração: escolhe melhor ação baseada na Q-table
            return self.get_best_action(state)
    
    def get_best_action(self, state):
        """Retorna a melhor ação para um estado (greedy policy)"""
        values = [self.Q[(state, a)] for a in range(self.n_actions)]
        return np.argmax(values)
    
    def update(self, state, action, reward, next_state, next_action, done):
        """Atualiza a Q-table usando SARSA"""
        if done:
            target = reward
        else:
            # SARSA: usa Q(next_state, next_action)
            target = reward + self.gamma * self.Q[(next_state, next_action)]
        
        # Atualização SARSA
        self.Q[(state, action)] += self.alpha * (target - self.Q[(state, action)])