module Env where

import Types (State(..), Action(..), Reward, StateVector)
import Numeric.LinearAlgebra (fromList)

gridSize :: Int
gridSize = 5

-- | Retorna o estado inicial do agente no canto superior esquerdo do grid
initState :: State
initState = State 0 0

-- | Verifica se o estado atual é o objetivo final (canto inferior direito)
isTerminal :: State -> Bool
isTerminal (State r c) = r == gridSize - 1 && c == gridSize - 1

-- | Executa uma ação no ambiente e retorna o novo estado, a recompensa e se alcançou o estado terminal
step :: State -> Action -> (State, Reward, Bool)
step s@(State r c) a =
  let nextState = case a of
        MoveUp    -> State (r - 1) c
        MoveDown  -> State (r + 1) c
        MoveLeft  -> State r (c - 1)
        MoveRight -> State r (c + 1)
      
      (State nr nc) = nextState
      outOfBounds = nr < 0 || nr >= gridSize || nc < 0 || nc >= gridSize
  in if outOfBounds
       then (s, -1.0, False)         -- Colisão com parede: recompensa -1.0, não move
       else if isTerminal nextState
              then (nextState, 1.0, True)   -- Alcançou objetivo: recompensa 1.0, terminal
              else (nextState, -0.01, False) -- Passo normal: recompensa -0.01

-- | Converte o estado atual em um vetor one-hot de 25 dimensões.
-- FEATURE ENGINEERING: Esta função é crucial para o DQL. Ela mapeia o espaço
-- de estados discretos (GridWorld) para um formato numérico contínuo.
-- É o ponto de conexão arquitetural exato que permite à Rede Neural (Etapa 4)
-- processar o estado e aproximar os Q-values na equação de Bellman.
encodeState :: State -> StateVector
encodeState (State r c) =
  let totalCells = gridSize * gridSize
      agentIdx = r * gridSize + c
      -- Gera o vetor one-hot: 1.0 na posição do agente, 0.0 nas demais
      oneHotList = [ if i == agentIdx then 1.0 else 0.0 | i <- [0 .. totalCells - 1] ]
  in fromList oneHotList
