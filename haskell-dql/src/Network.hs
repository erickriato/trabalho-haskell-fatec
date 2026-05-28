module Network where

import Numeric.LinearAlgebra (Matrix, Vector, (><), (#>), cmap, fromList)
import System.Random (StdGen, randomR)
import Types (StateVector)

-- | Arquitetura da Rede Neural que substituirá a Q-Table Clássica.
-- Input Layer: 25 neurônios (StateVector one-hot)
-- Hidden Layer: 64 neurônios (com ativação ReLU)
-- Output Layer: 4 neurônios (um Q-Value para cada Ação)
data Network = Network
  { weightsInput  :: Matrix Double -- ^ 64x25: da entrada para a camada oculta
  , biasHidden    :: Vector Double -- ^ 64 dimensões: bias da camada oculta
  , weightsHidden :: Matrix Double -- ^ 4x64: da camada oculta para a saída
  , biasOutput    :: Vector Double -- ^ 4 dimensões: bias da saída
  } deriving (Show)

-- | Função auxiliar pura para gerar N números aleatórios uniformes entre min e max
randomList :: Int -> (Double, Double) -> StdGen -> ([Double], StdGen)
randomList 0 _ gen = ([], gen)
randomList n bounds gen =
  let (v, gen') = randomR bounds gen
      (vs, gen'') = randomList (n - 1) bounds gen'
  in (v : vs, gen'')

-- | Inicializa os pesos e biases da Rede Neural de forma pura.
-- Utilizamos valores iniciais pequenos uniformemente distribuídos entre -0.1 e 0.1
-- para quebrar a simetria no início do treinamento e evitar gradientes explosivos.
initNetwork :: StdGen -> (Network, StdGen)
initNetwork gen0 =
  let wInSize = 64 * 25
      bHSize  = 64
      wHSize  = 4 * 64
      bOSize  = 4
      
      bounds = (-0.1, 0.1)
      
      (wInVals, gen1) = randomList wInSize bounds gen0
      (bHVals,  gen2) = randomList bHSize bounds gen1
      (wHVals,  gen3) = randomList wHSize bounds gen2
      (bOVals,  gen4) = randomList bOSize bounds gen3
      
      -- O operador (><) do hmatrix molda uma lista numa matriz (linhas >< colunas)
      -- E fromList cria um vetor a partir da lista
      net = Network
        { weightsInput  = (64 >< 25) wInVals
        , biasHidden    = fromList bHVals
        , weightsHidden = (4 >< 64) wHVals
        , biasOutput    = fromList bOVals
        }
  in (net, gen4)

-- | Função de Ativação ReLU (Rectified Linear Unit)
relu :: Vector Double -> Vector Double
relu = cmap (\x -> max 0 x)

-- | FORWARD PASS: O coração do Deep Q-Learning
-- Esta função é a substituta exata da função 'lookupQ' da Etapa 3 (Q-Table).
-- Dado o estado do ambiente (representado por um StateVector de dimensão 25),
-- ela retorna um vetor de dimensão 4 contendo a estimativa do Q-value para cada Ação.
-- Ao invés de uma tabela lookup discreta, a rede aproxima a função Q matematicamente.
forwardPass :: Network -> StateVector -> Vector Double
forwardPass net input =
  let -- Hidden Layer (Pre-ativação): Z1 = (W1 * X) + B1
      -- O operador (#>) faz a multiplicação matriz-vetor no hmatrix
      hiddenPre = (weightsInput net #> input) + biasHidden net
      
      -- Ativação ReLU: A1 = ReLU(Z1)
      hiddenAct = relu hiddenPre
      
      -- Output Layer (Estimativa dos Q-Values): Z2 = (W2 * A1) + B2
      outputQValues = (weightsHidden net #> hiddenAct) + biasOutput net
  in outputQValues
