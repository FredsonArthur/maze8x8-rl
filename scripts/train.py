import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from src.maze8x8_env import Maze8x8
from src.agents import QLearningAgent, SarsaAgent

def state_to_index(state):
    """Converte coordenada (x,y) para índice 0-63"""
    return state[0] * 8 + state[1]

def train_agent(env, agent, n_episodes, agent_type="QLearning"):
    """
    Treina um agente e retorna o histórico de recompensas e sucessos
    """
    rewards_per_episode = []
    success_per_episode = []
    
    for episode in range(n_episodes):
        state = env.reset()
        state_idx = state_to_index(state)
        done = False
        total_reward = 0
        steps = 0
        
        # Para SARSA, precisamos guardar a próxima ação
        if agent_type == "SARSA":
            action = agent.get_action(state_idx, training=True)
        
        while not done and steps < 200:  # limite de 200 passos
            if agent_type == "QLearning":
                # Q-Learning: escolhe ação
                action = agent.get_action(state_idx, training=True)
                # Executa ação
                next_state, reward, done, _ = env.step(action)
                next_state_idx = state_to_index(next_state)
                # Atualiza agente
                agent.update(state_idx, action, reward, next_state_idx, done)
                # Atualiza estado
                state_idx = next_state_idx
                
            elif agent_type == "SARSA":
                # SARSA: já temos action do passo anterior
                next_state, reward, done, _ = env.step(action)
                next_state_idx = state_to_index(next_state)
                
                # Escolhe próxima ação (se não terminou)
                if not done:
                    next_action = agent.get_action(next_state_idx, training=True)
                else:
                    next_action = None
                
                # Atualiza agente
                agent.update(state_idx, action, reward, next_state_idx, next_action, done)
                
                # Atualiza para próximo passo
                state_idx = next_state_idx
                action = next_action
            
            total_reward += reward
            steps += 1
        
        rewards_per_episode.append(total_reward)
        success_per_episode.append(1 if done else 0)
        
        # Mostra progresso a cada 500 episódios
        if (episode + 1) % 500 == 0:
            success_rate = np.mean(success_per_episode[-500:]) * 100
            print(f"{agent_type} - Episódio {episode+1}/{n_episodes} | "
                  f"Taxa de sucesso (últimos 500): {success_rate:.1f}% | "
                  f"Recompensa média: {np.mean(rewards_per_episode[-500:]):.2f}")
    
    return rewards_per_episode, success_per_episode

def smooth_curve(data, window=50):
    """Suaviza a curva usando média móvel"""
    smoothed = []
    for i in range(len(data)):
        start = max(0, i - window // 2)
        end = min(len(data), i + window // 2 + 1)
        smoothed.append(np.mean(data[start:end]))
    return smoothed

def plot_learning_curves(q_rewards, sarsa_rewards, window=50):
    """Plota as curvas de aprendizado"""
    plt.figure(figsize=(12, 5))
    
    # Suavizar curvas
    q_smooth = smooth_curve(q_rewards, window)
    sarsa_smooth = smooth_curve(sarsa_rewards, window)
    
    # Gráfico 1: Recompensa por episódio
    plt.subplot(1, 2, 1)
    plt.plot(q_smooth, label='Q-Learning', color='blue', alpha=0.8)
    plt.plot(sarsa_smooth, label='SARSA', color='red', alpha=0.8)
    plt.xlabel('Episódio')
    plt.ylabel('Recompensa Total')
    plt.title(f'Curvas de Aprendizado - Recompensa (média móvel {window})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/learning_curves.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\n📊 Gráfico salvo em: results/learning_curves.png")

def evaluate_agent(env, agent, n_episodes=100):
    """Avalia o agente com política greedy (sem exploração)"""
    successes = 0
    total_rewards = []
    
    for episode in range(n_episodes):
        state = env.reset()
        state_idx = state_to_index(state)
        done = False
        total_reward = 0
        
        while not done:
            action = agent.get_best_action(state_idx)  # política greedy
            next_state, reward, done, _ = env.step(action)
            state_idx = state_to_index(next_state)
            total_reward += reward
        
        if done:
            successes += 1
        total_rewards.append(total_reward)
    
    success_rate = (successes / n_episodes) * 100
    avg_reward = np.mean(total_rewards)
    
    return success_rate, avg_reward

# ==================== CONFIGURAÇÕES ====================
N_EPISODES = 3000  # número de episódios de treinamento
ALPHA = 0.1        # taxa de aprendizado
GAMMA = 0.99       # fator de desconto
EPSILON = 0.1      # exploração inicial

print("="*60)
print("🧠 TREINAMENTO - Maze 8x8")
print("="*60)
print(f"Parâmetros:")
print(f"  - Episódios: {N_EPISODES}")
print(f"  - Taxa de aprendizado (α): {ALPHA}")
print(f"  - Fator de desconto (γ): {GAMMA}")
print(f"  - Exploração (ε): {EPSILON}")
print("="*60)

# Criar ambiente
env = Maze8x8()

# Criar agentes com os mesmos parâmetros
q_agent = QLearningAgent(64, 4, alpha=ALPHA, gamma=GAMMA, epsilon=EPSILON)
sarsa_agent = SarsaAgent(64, 4, alpha=ALPHA, gamma=GAMMA, epsilon=EPSILON)

print("\n🏁 Iniciando treinamento Q-Learning...")
q_rewards, q_success = train_agent(env, q_agent, N_EPISODES, "QLearning")

print("\n🏁 Iniciando treinamento SARSA...")
sarsa_rewards, sarsa_success = train_agent(env, sarsa_agent, N_EPISODES, "SARSA")

# Plotar curvas
print("\n📈 Gerando gráficos...")
plot_learning_curves(q_rewards, sarsa_rewards)

# Avaliação final
print("\n" + "="*60)
print("📊 AVALIAÇÃO FINAL (100 episódios - política greedy)")
print("="*60)

q_success_rate, q_avg_reward = evaluate_agent(env, q_agent, 100)
sarsa_success_rate, sarsa_avg_reward = evaluate_agent(env, sarsa_agent, 100)

print(f"\n🏆 Q-LEARNING:")
print(f"   ✅ Taxa de sucesso: {q_success_rate:.1f}%")
print(f"   📈 Recompensa média: {q_avg_reward:.2f}")

print(f"\n🏆 SARSA:")
print(f"   ✅ Taxa de sucesso: {sarsa_success_rate:.1f}%")
print(f"   📈 Recompensa média: {sarsa_avg_reward:.2f}")

# Salvar resultados
import os
os.makedirs('results', exist_ok=True)

with open('results/evaluation_results.txt', 'w') as f:
    f.write("="*60 + "\n")
    f.write("AVALIAÇÃO MAZE 8x8 - Q-LEARNING vs SARSA\n")
    f.write("="*60 + "\n\n")
    f.write(f"Parâmetros de treinamento:\n")
    f.write(f"  - Episódios: {N_EPISODES}\n")
    f.write(f"  - α: {ALPHA}, γ: {GAMMA}, ε: {EPSILON}\n\n")
    f.write(f"Resultados (100 episódios greedy):\n")
    f.write(f"  Q-LEARNING: Taxa de sucesso = {q_success_rate:.1f}%\n")
    f.write(f"  SARSA:      Taxa de sucesso = {sarsa_success_rate:.1f}%\n\n")
    f.write(f"Diferença: {abs(q_success_rate - sarsa_success_rate):.1f}%\n")

print("\n💾 Resultados salvos em: results/evaluation_results.txt")