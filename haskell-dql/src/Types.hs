module Types where

import Data.Map.Strict (Map)
import Numeric.LinearAlgebra (Vector)

-- | Estado (posição no grid como dois inteiros: linha e coluna)
data State = State Int Int
  deriving (Show, Eq, Ord)

-- | Ação (movimentos possíveis)
data Action = MoveUp | MoveDown | MoveLeft | MoveRight
  deriving (Show, Eq, Ord, Enum, Bounded)

-- | Recompensa
type Reward = Float

-- | Q-Value
type QValue = Float

-- | Q-Table (para a etapa de andaime)
type QTable = Map (State, Action) QValue

-- | StateVector (vetor de estado para a Rede Neural via hmatrix)
type StateVector = Vector Double
