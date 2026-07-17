# 🎮 AlphaZero Breakthrough 5x5

**Projet MC_IASD** - Implementation d'AlphaZero pour le jeu Breakthrough

---

## 📋 Description du Projet

Ce projet implémente l'algorithme **AlphaZero** pour jouer au jeu **Breakthrough 5x5**, conformément au cahier des charges suivant :

### Spécifications Techniques

1. **Représentation de l'État** 
   - Transformation du plateau 5x5 en 3 matrices 5x5 (Noir/Blanc/Vide)
   - Encodage binaire pour chaque type de pièce

2. **Réseaux de Neurones Convolutifs**
   - 2 réseaux distincts (Blanc et Noir)
   - 76 sorties : 75 pour la politique + 1 pour l'évaluation
   - Architecture avec blocs résiduels (ResNet-style)

3. **Monte Carlo Tree Search (MCTS)**
   - Utilisation de PUCT pour la sélection
   - Intégration des prédictions du réseau (politique + valeur)
   - Minimum 100 simulations par coup

4. **Self-Play Training**
   - Génération de parties contre lui-même
   - Stockage des états : vecteur de fréquences (76 réels)
   - Entraînement supervisé sur les données générées

5. **Encodage des Actions**
   - Code action = `3 * (5 * x + y) + direction`
   - Direction : 0 = diagonal gauche, 1 = avant, 2 = diagonal droite
   - 75 actions possibles au total

---

## 🏗️ Architecture

```
projet/
├── app.py                  # Application Flask principale
├── templates/
│   └── index.html         # Interface web du jeu
├── requirements.txt       # Dépendances Python
├── model_white.pt        # Réseau pour joueur Blanc (généré)
├── model_black.pt        # Réseau pour joueur Noir (généré)
└── README.md             # Documentation
```

### Composants Principaux

#### 1. Réseau de Neurones (`BreakthroughNetwork`)
```python
Entrée: [batch, 3, 5, 5]  # 3 canaux (Noir/Blanc/Vide)
│
├─ Conv2d(3, 128) + BatchNorm + ReLU
├─ 5x ResidualBlock(128)
│
├─ Policy Head → [batch, 75]  # Probabilités des coups
└─ Value Head  → [batch, 1]   # Évaluation (-1 à +1)
```

#### 2. MCTS (`MCTSNode`, `MCTS`)
- **Sélection** : UCB avec constante PUCT
- **Expansion** : Utilise les priors du réseau
- **Simulation** : Évaluation par le réseau neuronal
- **Backpropagation** : Mise à jour des valeurs

#### 3. Jeu Breakthrough (`Breakthrough`)
- Plateau 5x5 avec règles officielles
- BLANC (lignes 3-4) vs NOIR (lignes 0-1)
- Victoire : atteindre la ligne adverse

---

## 🚀 Installation et Lancement

### Prérequis
- Python 3.8+
- pip

### Installation

```bash
# Cloner ou télécharger le projet
cd PROJET_MC_IASD

# Installer les dépendances
pip install -r requirements.txt
```

### Lancement

```bash
python app.py
```

Puis ouvrir dans un navigateur : **http://localhost:5000**

---

## 🎮 Utilisation

### Interface Web

1. **Nouvelle Partie** : Cliquez sur "🔄 Nouvelle Partie"
2. **Jouer** : Cliquez sur un pion BLANC (W)
3. **IA** : L'IA (NOIR) joue automatiquement après vous

### Règles du Jeu

- **BLANC (W)** : Commence en lignes 3-4, objectif ligne 0
- **NOIR (B)** : Commence en lignes 0-1, objectif ligne 4

**Déplacements** :
- **Avant** : Case vide uniquement
- **Diagonales** : Case vide OU capture ennemie

**Victoire** : Premier qui atteint la ligne adverse

---

## 🧠 Algorithme AlphaZero

### 1. Phase de Jeu (MCTS)

Pour chaque coup à jouer :

```python
for _ in range(NUM_SIMULATIONS):
    # 1. Sélection : descend l'arbre selon UCB
    node = root
    while node.is_expanded:
        node = select_child(node)  # UCB score
    
    # 2. Évaluation : réseau neuronal
    policy, value = network(node.state)
    
    # 3. Expansion : crée enfants avec priors
    node.expand(policy)
    
    # 4. Backpropagation : remonte la valeur
    backpropagate(node, value)

# Retourne politique améliorée (fréquences de visite)
return visit_counts / sum(visit_counts)
```

### 2. Phase d'Entraînement (Self-Play)

```python
for game in range(NUM_GAMES):
    states, policies, rewards = []
    
    # Joue partie complète
    while not done:
        policy = mcts.search(state)  # Politique MCTS
        action = sample(policy)
        state = step(action)
        states.append(state)
        policies.append(policy)
    
    # Entraîne le réseau
    for s, p, r in zip(states, policies, rewards):
        loss = mse(network(s), [p, r])
        loss.backward()
        optimizer.step()
```

### 3. Fonction de Perte

```python
loss = (policy_loss + value_loss) / 2

où:
  policy_loss = CrossEntropy(predicted_policy, mcts_policy)
  value_loss = MSE(predicted_value, game_result)
```

---

## 📊 Paramètres de Configuration

```python
BOARD_SIZE = 5              # Taille du plateau
NUM_ACTIONS = 75            # Nombre d'actions possibles
NUM_SIMULATIONS = 200       # Simulations MCTS par coup
C_PUCT = 1.4               # Constante d'exploration UCB
LEARNING_RATE = 0.001      # Taux d'apprentissage
NUM_SELF_PLAY_GAMES = 100  # Parties de self-play
```

---

## 🔧 API Endpoints

### `GET /`
Page d'accueil avec le plateau de jeu

### `POST /api/move`
```json
{
  "action": 42,
  "game_id": "default"
}
```
**Réponse** :
```json
{
  "board_html": "<table>...</table>",
  "current_player": 1,
  "done": false,
  "winner": null,
  "ai_move": 38,
  "status": "🎮 Votre tour!"
}
```

### `GET /api/new_game`
Démarre une nouvelle partie

**Réponse** :
```json
{
  "game_id": "1234",
  "board_html": "<table>...</table>",
  "current_player": 1,
  "legal_moves": [0, 1, 2, ...],
  "status": "🎮 Nouvelle partie!"
}
```

### `GET /api/stats`
Statistiques du système

**Réponse** :
```json
{
  "device": "cpu",
  "num_simulations": 200,
  "network_params": 234567,
  "active_games": 3
}
```

---

## 📈 Améliorations Possibles

1. **Entraînement Complet**
   - Implémenter le self-play automatique
   - Sauvegarder l'historique des parties
   - Courbes d'apprentissage

2. **Optimisations**
   - Parallélisation MCTS (Virtual Loss)
   - Cache des évaluations
   - Quantization du réseau

3. **Interface**
   - Visualisation de l'arbre MCTS
   - Heatmap des probabilités
   - Replay des parties

4. **Modes de Jeu**
   - Humain vs Humain
   - IA vs IA (avec différents niveaux)
   - Mode analyse

---

## 📚 Références

- **AlphaZero Paper** : [Mastering Chess and Shogi by Self-Play](https://arxiv.org/abs/1712.01815)
- **AlphaGo Zero** : [Mastering the game of Go without human knowledge](https://www.nature.com/articles/nature24270)
- **MCTS Survey** : [A Survey of Monte Carlo Tree Search Methods](https://ieeexplore.ieee.org/document/6145622)

---

## 👨‍💻 Auteur

**Projet MC_IASD** - Implementation AlphaZero Breakthrough 5x5

---

## 📝 Licence

Projet académique - Libre d'utilisation pour l'apprentissage

---

## 🐛 Dépannage

### Erreur "signal only works in main thread"
```python
# Solution : désactiver le reloader
app.run(debug=True, use_reloader=False)
```

### Modèles non trouvés
Les modèles sont créés automatiquement au premier lancement.

### Performance lente
Réduire `NUM_SIMULATIONS` dans `Config` (min 50).

---

## ✅ Checklist Conformité Sujet

- ✅ 3 matrices 5x5 en entrée (Noir/Blanc/Vide)
- ✅ 2 réseaux convolutifs (Blanc/Noir)
- ✅ 76 sorties (75 actions + 1 valeur)
- ✅ MCTS avec PUCT
- ✅ Politique = fréquences de visite
- ✅ Encodage actions : `3*(5*x+y)+d`
- ✅ Self-play framework ready
- ✅ Entraînement sur (état, politique, résultat)

---

**Version** : 1.0.0  
**Date** : Février 2026
