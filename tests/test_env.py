"""
Testes automatizados para o ambiente Maze8x8
Disciplina: Tópicos Especiais em Inteligência Artificial - IFPE

Executar com: python tests/test_env.py
"""

import sys
import os

# Adiciona o diretório raiz ao path para importar os módulos
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from src.maze8x8_env import Maze8x8

def test_criacao_ambiente():
    """Testa se o ambiente é criado com as configurações corretas"""
    print("\n[TESTE 1] Criando ambiente...")
    env = Maze8x8()
    
    assert env.size == 8, f"Erro: size deveria ser 8, mas é {env.size}"
    assert env.start_state == (0, 0), f"Erro: start_state deveria ser (0,0), mas é {env.start_state}"
    assert env.goal_state == (7, 7), f"Erro: goal_state deveria ser (7,7), mas é {env.goal_state}"
    assert len(env.holes) == 22, f"Erro: deveria ter 22 paredes, mas tem {len(env.holes)}"
    assert env.n_actions == 4, f"Erro: deveria ter 4 ações, mas tem {env.n_actions}"
    
    print("✅ OK: Ambiente criado corretamente")
    return True

def test_reset():
    """Testa se o reset volta ao estado inicial"""
    print("\n[TESTE 2] Testando reset...")
    env = Maze8x8()
    
    # Modifica o estado
    env.state = (5, 5)
    
    # Reseta
    state = env.reset()
    
    assert state == (0, 0), f"Erro: reset deveria retornar (0,0), mas retornou {state}"
    assert env.state == (0, 0), f"Erro: estado após reset deveria ser (0,0), mas é {env.state}"
    
    print("✅ OK: Reset funcionando")
    return True

def test_movimento_normal():
    """Testa movimentos básicos sem paredes"""
    print("\n[TESTE 3] Testando movimentos básicos...")
    env = Maze8x8()
    env.reset()
    
    # Testa movimento para BAIXO (ação 1)
    next_state, reward, done, _ = env.step(1)
    assert next_state == (1, 0), f"Erro: mover baixo de (0,0) deveria ir para (1,0), mas foi para {next_state}"
    assert reward == -1, f"Erro: recompensa deveria ser -1, mas foi {reward}"
    assert done == False, "Erro: não deveria ter terminado o episódio"
    
    # Testa movimento para DIREITA (ação 3)
    env.reset()
    next_state, reward, done, _ = env.step(3)
    assert next_state == (0, 1), f"Erro: mover direita de (0,0) deveria ir para (0,1), mas foi para {next_state}"
    
    print("✅ OK: Movimentos básicos funcionando")
    return True

def test_paredes():
    """Testa se as paredes bloqueiam o movimento"""
    print("\n[TESTE 4] Testando bloqueio por paredes...")
    env = Maze8x8()
    env.reset()
    
    # (0,5) é uma parede
    for _ in range(5):
        env.step(3)  # vai para direita até (0,4)
    
    pos_antes = env.state
    next_state, _, _, _ = env.step(3)  # tenta ir para (0,5)
    
    assert next_state == pos_antes, f"Erro: parede em (0,5) deveria bloquear, mas moveu de {pos_antes} para {next_state}"
    
    print("✅ OK: Paredes bloqueiam movimento")
    return True

def test_bordas():
    """Testa se as bordas limitam o movimento"""
    print("\n[TESTE 5] Testando limites do grid...")
    env = Maze8x8()
    env.reset()
    
    # Tenta sair pela borda superior
    pos_antes = env.state
    next_state, _, _, _ = env.step(0)  # cima
    assert next_state == pos_antes, f"Erro: deveria ficar em {pos_antes} ao tentar sair da grade, mas foi para {next_state}"
    
    # Vai para o canto inferior direito
    env.state = (7, 6)
    pos_antes = env.state
    next_state, _, _, _ = env.step(3)  # direita para (7,7) - é objetivo, então move
    assert next_state == (7, 7), f"Erro: deveria ir para (7,7), mas foi para {next_state}"
    
    # Tenta sair pela borda direita a partir do objetivo
    env.state = (7, 7)
    pos_antes = env.state
    next_state, _, _, _ = env.step(3)  # direita
    assert next_state == pos_antes, f"Erro: após objetivo, deveria ficar em (7,7), mas foi para {next_state}"
    
    print("✅ OK: Bordas limitam movimento")
    return True

def test_objetivo():
    """Testa chegada ao objetivo"""
    print("\n[TESTE 6] Testando objetivo e recompensa...")
    env = Maze8x8()
    
    # Coloca o agente ao lado do objetivo
    env.state = (7, 6)
    next_state, reward, done, _ = env.step(3)  # move para direita (7,7)
    
    assert next_state == (7, 7), f"Erro: deveria ir para (7,7), mas foi para {next_state}"
    assert reward == 10, f"Erro: recompensa ao chegar no objetivo deveria ser +10, mas foi {reward}"
    assert done == True, "Erro: episódio deveria terminar quando chega ao objetivo"
    
    print("✅ OK: Objetivo e recompensa corretos")
    return True

def test_render():
    """Testa se o método render executa sem erros"""
    print("\n[TESTE 7] Testando renderização...")
    env = Maze8x8()
    try:
        env.render()
        print("✅ OK: Renderização executou sem erros")
        return True
    except Exception as e:
        print(f"❌ ERRO: Renderização falhou - {e}")
        return False

def test_acoes_validas():
    """Testa se o método get_valid_actions funciona"""
    print("\n[TESTE 8] Testando ações válidas...")
    env = Maze8x8()
    actions = env.get_valid_actions()
    
    assert actions == [0, 1, 2, 3], f"Erro: ações deveriam ser [0,1,2,3], mas são {actions}"
    
    print("✅ OK: Ações válidas retornadas corretamente")
    return True

def test_estado_terminal():
    """Testa verificação de estado terminal"""
    print("\n[TESTE 9] Testando estado terminal...")
    env = Maze8x8()
    
    assert not env.is_terminal((0, 0)), "Erro: (0,0) não deveria ser terminal"
    assert env.is_terminal((7, 7)), "Erro: (7,7) deveria ser terminal"
    
    print("✅ OK: Verificação de estado terminal correta")
    return True

def executar_todos_testes():
    """Executa todos os testes e reporta resultados"""
    print("="*60)
    print("🧪 VALIDAÇÃO DO AMBIENTE MAZE 8x8")
    print("="*60)
    
    testes = [
        test_criacao_ambiente,
        test_reset,
        test_movimento_normal,
        test_paredes,
        test_bordas,
        test_objetivo,
        test_render,
        test_acoes_validas,
        test_estado_terminal
    ]
    
    passaram = 0
    falharam = 0
    
    for teste in testes:
        try:
            if teste():
                passaram += 1
        except AssertionError as e:
            print(f"❌ FALHOU: {e}")
            falharam += 1
        except Exception as e:
            print(f"❌ ERRO INESPERADO: {e}")
            falharam += 1
    
    print("\n" + "="*60)
    print("📊 RESUMO DOS TESTES")
    print("="*60)
    print(f"✅ Testes passaram: {passaram}")
    print(f"❌ Testes falharam: {falharam}")
    
    if falharam == 0:
        print("\n🎉 PARABÉNS! Todos os testes passaram!")
        print("   O ambiente Maze8x8 está funcionando corretamente.")
    else:
        print(f"\n⚠️ ATENÇÃO: {falharam} teste(s) falharam. Revise o código.")
    
    print("="*60)
    return falharam == 0

if __name__ == "__main__":
    sucesso = executar_todos_testes()
    sys.exit(0 if sucesso else 1)