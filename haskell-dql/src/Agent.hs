module Agent where

import Types (Action(..), StateVector)
import Network (Network, forwardPass)
import Backprop (computeTarget, updateWeights)
import ReplayBuffer (Experience)
import System.Random (StdGen, randomR)
import Numeric.LinearAlgebra (maxIndex)
import qualified Config

-- | POLÍTICA EPSILON-GREEDY
-- Esta função cumpre exatamente o mesmo papel de 'bestAction' do módulo QTable.hs,
-- mas consultando a Rede Neural (via forwardPass) para estimar os valores, 
-- em vez de buscar em um dicionário de estados discretos.
-- Com probabilidade 'epsilon', o agente explora (ação aleatória).
-- Com probabilidade '1 - epsilon', o agente explora o melhor Q-Value.
-- Nota: Omitimos o parâmetro Config da assinatura pois importamos as variáveis do módulo Config.hs.
selectAction :: Network -> StateVector -> Double -> StdGen -> (Action, StdGen)
selectAction net state epsilon gen =
  let (p, gen') = randomR (0.0, 1.0) gen
  in if p < epsilon
       then -- Exploração: ação aleatória
            let (actIdx, gen'') = randomR (0, 3) gen'
            in (toEnum actIdx, gen'')
       else -- Explotação: ação com maior Q-Value estimado pela rede
            let qValues = forwardPass net state
                bestIdx = maxIndex qValues
            in (toEnum bestIdx, gen')

-- | Aplica o decaimento do Epsilon ao fim de cada episódio para reduzir a exploração,
-- garantindo que não fique abaixo do Epsilon mínimo definido nas configurações.
decayEpsilon :: Double -> Double
decayEpsilon currentEpsilon =
  max Config.epsilonMin (currentEpsilon * Config.epsilonDecay)

-- | Loop de treinamento do Mini-Batch extraído do Replay Buffer.
-- Realiza um 'foldl' sobre a lista de experiências. Para cada transição, 
-- calcula o alvo da Equação de Bellman e ajusta os pesos da Rede Neural via gradiente descendente.
trainOnBatch :: Network -> [Experience] -> Network
trainOnBatch initialNet batch =
  foldl trainStep initialNet batch
  where
    trainStep :: Network -> Experience -> Network
    trainStep currentNet (s, a, r, nextS, isTerm) =
      let target = computeTarget currentNet r nextS isTerm
      in updateWeights currentNet s a target
