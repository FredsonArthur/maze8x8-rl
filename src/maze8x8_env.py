import numpy as np

class Maze8x8:
    """
    Ambiente Maze 8x8 com paredes (holes) definidas pelo gist.
    Início: (0,0)
    Objetivo: (7,7)
    Recompensa: +10 no objetivo, -1 caso contrário
    """
    
    def __init__(self):
        # Grid 8x8
        self.size = 8
        self.start_state = (0, 0)
        self.goal_state = (7, 7)
        
        # Matriz de paredes (1 = parede, 0 = livre) - DO GIST
        walls_matrix = [
            [0, 0, 0, 0, 0, 1, 0, 0],  # linha 0
            [0, 1, 1, 1, 0, 1, 0, 1],  # linha 1
            [0, 0, 0, 0, 0, 0, 0, 0],  # linha 2
            [1, 1, 0, 1, 1, 1, 1, 0],  # linha 3
            [0, 0, 0, 0, 0, 0, 0, 0],  # linha 4
            [0, 1, 1, 1, 1, 1, 0, 1],  # linha 5
            [0, 0, 0, 0, 0, 0, 0, 0],  # linha 6
            [0, 1, 0, 1, 1, 0, 1, 0]   # linha 7
        ]
        
        # Converte a matriz em lista de coordenadas de paredes
        self.holes = []
        for i in range(self.size):
            for j in range(self.size):
                if walls_matrix[i][j] == 1:
                    self.holes.append((i, j))
        
        # Ações: 0=cima, 1=baixo, 2=esquerda, 3=direita
        self.actions = [0, 1, 2, 3]
        self.n_actions = 4
        
        # Estado atual
        self.state = None
        
        # Resetar ambiente
        self.reset()
    
    def reset(self):
        """Reinicia o ambiente para o estado inicial"""
        self.state = self.start_state
        return self.state
    
    def step(self, action):
        """
        Executa uma ação no ambiente.
        Retorna: (next_state, reward, done, info)
        """
        x, y = self.state
        
        # Calcula próxima posição baseado na ação
        if action == 0:  # cima
            next_state = (max(0, x-1), y)
        elif action == 1:  # baixo
            next_state = (min(self.size-1, x+1), y)
        elif action == 2:  # esquerda
            next_state = (x, max(0, y-1))
        elif action == 3:  # direita
            next_state = (x, min(self.size-1, y+1))
        else:
            raise ValueError(f"Ação inválida: {action}")
        
        # Verifica se a próxima posição é uma parede (hole)
        if next_state in self.holes:
            # Se for parede, não move
            next_state = self.state
        
        # Atualiza estado
        self.state = next_state
        
        # Calcula recompensa
        if self.state == self.goal_state:
            reward = 10
            done = True
        else:
            reward = -1
            done = False
        
        return self.state, reward, done, {}
    
    def render(self):
        """Imprime o mapa do ambiente"""
        grid = [[' ' for _ in range(self.size)] for _ in range(self.size)]
        
        # Marca as paredes
        for (x, y) in self.holes:
            grid[x][y] = '█'
        
        # Marca posições especiais
        x, y = self.start_state
        grid[x][y] = 'S'
        
        x, y = self.goal_state
        grid[x][y] = 'G'
        
        # Marca posição atual do agente (se não for S ou G)
        x, y = self.state
        if self.state != self.start_state and self.state != self.goal_state:
            grid[x][y] = 'A'
        
        # Imprime o grid
        print('+' + '---+' * self.size)
        for i in range(self.size):
            linha = '|'
            for j in range(self.size):
                linha += f' {grid[i][j]} |'
            print(linha)
            print('+' + '---+' * self.size)
    
    def get_valid_actions(self, state=None):
        """Retorna ações válidas (todas são válidas no Maze)"""
        return self.actions
    
    def is_terminal(self, state):
        """Verifica se o estado é terminal (objetivo)"""
        return state == self.goal_state