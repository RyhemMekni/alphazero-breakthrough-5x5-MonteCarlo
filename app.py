"""
AlphaZero Breakthrough 5x5 - Projet MC_IASD
Implementation complète selon le cahier des charges
"""

from flask import Flask, render_template, request, jsonify
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import math
from collections import defaultdict
import copy

app = Flask(__name__)
app.config['SECRET_KEY'] = 'breakthrough_alphazero_2024'

# ========== CONFIGURATION ==========
class Config:
    BOARD_SIZE = 5
    NUM_ACTIONS = 75  # 75 coups possibles
    NUM_SIMULATIONS = 200  # Nombre de simulations MCTS
    C_PUCT = 1.4  # Constante d'exploration
    LEARNING_RATE = 0.001
    NUM_SELF_PLAY_GAMES = 100
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CFG = Config()

# ========== RÉSEAU DE NEURONES CONVOLUTIF ==========
class ResidualBlock(nn.Module):
    """Bloc résiduel pour le réseau convolutif"""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)

class BreakthroughNetwork(nn.Module):
    """
    Réseau convolutif avec 76 sorties:
    - 75 pour la politique (probabilité de chaque coup)
    - 1 pour la valeur (évaluation de position: 1.0 si blanc gagne, 0.0 sinon)
    
    Entrée: 3 matrices 5x5 (Noir/Blanc/Vide)
    """
    def __init__(self):
        super().__init__()
        # Couche d'entrée
        self.conv_input = nn.Conv2d(3, 128, kernel_size=3, padding=1)
        self.bn_input = nn.BatchNorm2d(128)
        
        # Blocs résiduels
        self.res_blocks = nn.ModuleList([ResidualBlock(128) for _ in range(5)])
        
        # Tête de politique (75 sorties pour les coups)
        self.policy_conv = nn.Conv2d(128, 32, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * 5 * 5, 75)
        
        # Tête de valeur (1 sortie pour l'évaluation)
        self.value_conv = nn.Conv2d(128, 8, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(8)
        self.value_fc1 = nn.Linear(8 * 5 * 5, 64)
        self.value_fc2 = nn.Linear(64, 1)
    
    def forward(self, x):
        # Backbone convolutif
        x = F.relu(self.bn_input(self.conv_input(x)))
        for block in self.res_blocks:
            x = block(x)
        
        # Politique
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(-1, 32 * 5 * 5)
        policy = self.policy_fc(policy)
        
        # Valeur
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(-1, 8 * 5 * 5)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))
        
        return policy, value

# ========== JEU BREAKTHROUGH ==========
class Breakthrough:
    """
    Jeu Breakthrough 5x5
    - BLANC (1): commence en lignes 3-4, objectif: atteindre ligne 0
    - NOIR (-1): commence en lignes 0-1, objectif: atteindre ligne 4
    """
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Initialise le plateau"""
        self.board = np.zeros((5, 5), dtype=np.int8)
        # NOIR en haut (lignes 0-1)
        for j in range(5):
            self.board[0, j] = -1
            self.board[1, j] = -1
        # BLANC en bas (lignes 3-4)
        for j in range(5):
            self.board[3, j] = 1
            self.board[4, j] = 1
        
        self.current_player = 1  # BLANC commence
        self.move_count = 0
    
    def clone(self):
        """Clone l'état du jeu"""
        new_game = Breakthrough()
        new_game.board = self.board.copy()
        new_game.current_player = self.current_player
        new_game.move_count = self.move_count
        return new_game
    
    def get_legal_actions(self):
        """
        Retourne les actions légales (codes 0-74)
        Code action = 3 * (5 * x + y) + direction
        où direction: 0=diagonal gauche, 1=avant, 2=diagonal droite
        """
        actions = []
        player = self.current_player
        
        # Directions selon le joueur
        if player == 1:  # BLANC monte (dx = -1)
            dirs = [(-1, -1), (-1, 0), (-1, 1)]
        else:  # NOIR descend (dx = +1)
            dirs = [(1, -1), (1, 0), (1, 1)]
        
        for x in range(5):
            for y in range(5):
                if self.board[x, y] != player:
                    continue
                
                for d, (dx, dy) in enumerate(dirs):
                    nx, ny = x + dx, y + dy
                    
                    # Vérification limites
                    if not (0 <= nx < 5 and 0 <= ny < 5):
                        continue
                    
                    target = self.board[nx, ny]
                    
                    # Règles de déplacement:
                    # - Avant (d=1): case vide uniquement
                    # - Diagonales (d=0,2): case vide OU ennemi
                    if d == 1:  # Avant
                        if target == 0:
                            actions.append(3 * (5 * x + y) + d)
                    else:  # Diagonales
                        if target == 0 or target == -player:
                            actions.append(3 * (5 * x + y) + d)
        
        return sorted(set(actions))
    
    def step(self, action):
        """
        Exécute une action
        Retourne: (done, winner)
        """
        legal_actions = self.get_legal_actions()
        
        # Gestion action illégale
        if action not in legal_actions:
            if not legal_actions:
                return True, -self.current_player  # Joueur actuel perd
            action = legal_actions[0]
        
        # Décode l'action
        d = action % 3
        pos = action // 3
        x, y = divmod(pos, 5)
        
        # Directions
        if self.current_player == 1:
            dirs = [(-1, -1), (-1, 0), (-1, 1)]
        else:
            dirs = [(1, -1), (1, 0), (1, 1)]
        
        dx, dy = dirs[d]
        nx, ny = x + dx, y + dy
        
        # Exécute le mouvement
        self.board[nx, ny] = self.current_player
        self.board[x, y] = 0
        
        # Change de joueur
        self.current_player *= -1
        self.move_count += 1
        
        # Vérifie victoire
        winner = self.check_winner()
        done = winner != 0 or self.move_count >= 100  # Max 100 coups
        
        return done, winner
    
    def check_winner(self):
        """
        Vérifie les conditions de victoire
        BLANC gagne si atteint ligne 0
        NOIR gagne si atteint ligne 4
        """
        if np.any(self.board[0, :] == 1):  # BLANC en ligne 0
            return 1
        if np.any(self.board[4, :] == -1):  # NOIR en ligne 4
            return -1
        return 0
    
    def board_to_input(self):
        """
        Convertit le plateau en 3 matrices 5x5 (Noir/Blanc/Vide)
        Format: [1, 3, 5, 5] pour le réseau
        """
        white = (self.board == 1).astype(np.float32)
        black = (self.board == -1).astype(np.float32)
        empty = (self.board == 0).astype(np.float32)
        
        # Inverse la perspective pour le joueur noir
        if self.current_player == -1:
            white, black = black, white
            white = np.flip(white, axis=0).copy()
            black = np.flip(black, axis=0).copy()
            empty = np.flip(empty, axis=0).copy()
        
        return np.stack([white, black, empty], axis=0)[None, ...]
    
    def to_html(self):
        """Génère le HTML du plateau"""
        chars = {1: 'W', -1: 'B', 0: '·'}
        rows = []
        for i, row in enumerate(self.board):
            cells = []
            for j, cell in enumerate(row):
                cls = 'cell'
                if cell == 1:
                    cls += ' white'
                elif cell == -1:
                    cls += ' black'
                else:
                    cls += ' empty'
                
                cells.append(f'<td class="{cls}" data-x="{i}" data-y="{j}">{chars[cell]}</td>')
            rows.append(f'<tr>{"".join(cells)}</tr>')
        
        return f'<table class="board">{"".join(rows)}</table>'

# ========== MCTS (Monte Carlo Tree Search) ==========
class MCTSNode:
    """Nœud de l'arbre MCTS"""
    def __init__(self, game_state, parent=None, action=None, prior=0):
        self.game_state = game_state
        self.parent = parent
        self.action = action
        self.prior = prior
        
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0
        self.is_expanded = False
    
    def value(self):
        """Valeur moyenne du nœud"""
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count
    
    def select_child(self):
        """Sélectionne le meilleur enfant selon UCB"""
        best_score = -float('inf')
        best_child = None
        
        for child in self.children.values():
            # UCB score
            if child.visit_count == 0:
                ucb_score = float('inf')
            else:
                exploitation = child.value()
                exploration = CFG.C_PUCT * child.prior * math.sqrt(self.visit_count) / (1 + child.visit_count)
                ucb_score = exploitation + exploration
            
            if ucb_score > best_score:
                best_score = ucb_score
                best_child = child
        
        return best_child
    
    def expand(self, policy_probs):
        """Étend le nœud avec les actions légales"""
        legal_actions = self.game_state.get_legal_actions()
        
        for action in legal_actions:
            if action not in self.children:
                # Clone l'état et exécute l'action
                new_state = self.game_state.clone()
                new_state.step(action)
                
                # Crée l'enfant
                prior = policy_probs[action] if action < len(policy_probs) else 1.0 / len(legal_actions)
                self.children[action] = MCTSNode(new_state, parent=self, action=action, prior=prior)
        
        self.is_expanded = True
    
    def backpropagate(self, value):
        """Propage la valeur vers la racine"""
        self.visit_count += 1
        self.value_sum += value
        
        if self.parent:
            self.parent.backpropagate(-value)  # Inverse pour l'adversaire

class MCTS:
    """Monte Carlo Tree Search avec réseau de neurones"""
    def __init__(self, network):
        self.network = network
        self.network.eval()
    
    def search(self, game_state, num_simulations=CFG.NUM_SIMULATIONS):
        """
        Effectue num_simulations simulations MCTS
        Retourne: vecteur de fréquences de visite (politique améliorée)
        """
        root = MCTSNode(game_state)
        
        for _ in range(num_simulations):
            node = root
            search_path = [node]
            
            # Sélection
            while node.is_expanded and node.children:
                node = node.select_child()
                search_path.append(node)
            
            # Évaluation
            done, winner = node.game_state.check_winner() != 0, node.game_state.check_winner()
            
            if done:
                # Position terminale
                value = winner
            else:
                # Évalue avec le réseau
                board_input = torch.FloatTensor(node.game_state.board_to_input()).to(CFG.device)
                
                with torch.no_grad():
                    policy_logits, value = self.network(board_input)
                    policy_probs = F.softmax(policy_logits, dim=1).cpu().numpy()[0]
                    value = value.item()
                
                # Expansion
                node.expand(policy_probs)
            
            # Backpropagation
            for n in reversed(search_path):
                n.backpropagate(value)
                value = -value
        
        # Retourne les fréquences de visite
        action_visits = np.zeros(75, dtype=np.float32)
        for action, child in root.children.items():
            action_visits[action] = child.visit_count
        
        # Normalise
        if action_visits.sum() > 0:
            action_visits = action_visits / action_visits.sum()
        
        return action_visits

# ========== MODÈLES GLOBAUX ==========
network_white = None
network_black = None
mcts_white = None
mcts_black = None
games = {}

def load_models():
    """Charge ou initialise les modèles"""
    global network_white, network_black, mcts_white, mcts_black
    
    try:
        network_white = torch.load('model_white.pt', map_location=CFG.device)
        network_black = torch.load('model_black.pt', map_location=CFG.device)
        print("✅ Modèles chargés depuis les fichiers")
    except:
        print("⚠️ Création de nouveaux modèles...")
        network_white = BreakthroughNetwork().to(CFG.device)
        network_black = BreakthroughNetwork().to(CFG.device)
        torch.save(network_white, 'model_white.pt')
        torch.save(network_black, 'model_black.pt')
        print("💾 Modèles sauvegardés")
    
    mcts_white = MCTS(network_white)
    mcts_black = MCTS(network_black)

# ========== ROUTES FLASK ==========
@app.route('/')
def index():
    """Page d'accueil"""
    load_models()
    game = Breakthrough()
    return render_template('index.html', 
                         board_html=game.to_html(),
                         current_player=game.current_player,
                         status="🎮 Jouez BLANC! Cliquez sur un pion blanc")

@app.route('/api/move', methods=['POST'])
def api_move():
    """Traite un coup du joueur et fait jouer l'IA"""
    try:
        data = request.json
        game_id = data.get('game_id', 'default')
        action = int(data['action'])
        
        # Récupère ou crée la partie
        if game_id not in games:
            games[game_id] = Breakthrough()
        game = games[game_id]
        
        print(f"👤 Joueur: action={action}, player={game.current_player}")
        
        # Vérifie si l'action est légale
        legal_actions = game.get_legal_actions()
        if action not in legal_actions:
            return jsonify({
                'error': f'Coup illégal! Actions légales: {legal_actions[:10]}...',
                'legal_moves': legal_actions,
                'board_html': game.to_html(),
                'current_player': int(game.current_player)
            })
        
        # Exécute le coup du joueur
        done, winner = game.step(action)
        print(f"Après joueur: done={done}, winner={winner}")
        
        # IA joue si la partie continue
        ai_action = None
        if not done:
            # Sélectionne le réseau approprié
            mcts = mcts_black if game.current_player == -1 else mcts_white
            
            # MCTS pour trouver le meilleur coup
            policy = mcts.search(game, num_simulations=100)
            legal_actions = game.get_legal_actions()
            
            # Sélectionne l'action la plus visitée parmi les légales
            legal_policy = np.zeros_like(policy)
            for a in legal_actions:
                legal_policy[a] = policy[a]
            
            if legal_policy.sum() > 0:
                ai_action = np.argmax(legal_policy)
            else:
                ai_action = random.choice(legal_actions)
            
            done, winner = game.step(ai_action)
            print(f"🤖 IA: action={ai_action}, done={done}, winner={winner}")
        
        # Prépare la réponse
        status_text = {
            0: '🎮 Votre tour!' if not done else '⚔️ Égalité!',
            1: '🎉 BLANC GAGNE!',
            -1: '🤖 NOIR GAGNE!'
        }.get(winner, '🎮 Votre tour!')
        
        return jsonify({
            'board_html': game.to_html(),
            'current_player': int(game.current_player),
            'done': done,
            'winner': 'Blanc' if winner == 1 else 'Noir' if winner == -1 else None,
            'status': status_text,
            'ai_move': int(ai_action) if ai_action is not None else None,
            'legal_moves': game.get_legal_actions()
        })
    
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/new_game')
def new_game():
    """Démarre une nouvelle partie"""
    game_id = str(random.randint(1000, 9999))
    games[game_id] = Breakthrough()
    game = games[game_id]
    
    return jsonify({
        'game_id': game_id,
        'board_html': game.to_html(),
        'current_player': game.current_player,
        'legal_moves': game.get_legal_actions(),
        'status': '🎮 Nouvelle partie! Jouez BLANC'
    })

@app.route('/api/stats')
def stats():
    """Retourne les statistiques du réseau"""
    return jsonify({
        'device': str(CFG.device),
        'num_simulations': CFG.NUM_SIMULATIONS,
        'network_params': sum(p.numel() for p in network_white.parameters()),
        'active_games': len(games)
    })

if __name__ == '__main__':
    print("🚀 AlphaZero Breakthrough 5x5")
    print(f" Device: {CFG.device}")
    print(f"🎯 MCTS Simulations: {CFG.NUM_SIMULATIONS}")
    print(" Accès: http://localhost:5000")
    print("-" * 50)
    
    load_models()
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False, threaded=True)
