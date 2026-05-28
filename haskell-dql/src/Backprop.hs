module Backprop where

import Types (Action(..), Reward, StateVector)
import Network (Network(..), forwardPass, relu)
import Numeric.LinearAlgebra (maxElement, fromList, outer, tr, cmap, scale, toList, (#>))
import qualified Config

-- | Calcula o Q-value alvo (Target) para a Equação de Bellman
-- BELLMAN EQUATION:
-- target = reward + gamma * max(Q(nextS, a'))   (se não for terminal)
-- target = reward                               (se for terminal)
computeTarget :: Network -> Reward -> StateVector -> Bool -> Double
computeTarget net r nextS isTerminal =
  let rewardD = realToFrac r
  in if isTerminal
        then rewardD
        else rewardD + Config.gamma * maxElement (forwardPass net nextS)

-- | Calcula o erro (Loss) usando o Erro Quadrático Médio (MSE).
-- Loss = (predicted - target)^2
computeLoss :: Double -> Double -> Double
computeLoss predicted target = (predicted - target) ^ 2

-- | ATUALIZAÇÃO DE PESOS (Backpropagation via Gradiente Descendente)
-- DIFERENÇA CRUCIAL DA ETAPA 3 (Q-TABLE):
-- Na Q-Table Clássica, ao atualizar Q(s,a), apenas aquela célula específica muda.
-- Na Rede Neural, os pesos (weightsInput, weightsHidden, etc) são globais e 
-- compartilhados. Ao ajustá-los via backpropagation para reduzir o erro em Q(s,a),
-- nós também alteramos (generalizamos) a estimativa de Q-values para TODOS os outros
-- estados e ações simultaneamente. É isso que permite ao agente aprender mais rápido,
-- mas também pode causar instabilidade (daí a necessidade do Replay Buffer na Etapa 6).
updateWeights :: Network -> StateVector -> Action -> Double -> Network
updateWeights net input action target =
  let alpha = Config.alpha
      
      -- 1. Forward Pass (Re-computando para obter as ativações intermediárias)
      z1 = (weightsInput net #> input) + biasHidden net
      a1 = relu z1
      z2 = (weightsHidden net #> a1) + biasOutput net
      
      -- Pega o valor predito apenas para a ação que foi tomada
      predicted = (toList z2) !! fromEnum action
      
      -- Gradiente do Erro (delta = predicted - target)
      delta = predicted - target
      
      -- 2. Backpropagation
      -- Vetor de erro da camada de saída: zero para as ações não tomadas, 'delta' para a tomada.
      dZ2List = [ if i == fromEnum action then delta else 0.0 | i <- [0 .. 3] ]
      dZ2 = fromList dZ2List
      
      -- Gradientes da camada Oculta -> Saída
      dW2 = dZ2 `outer` a1
      dB2 = dZ2
      
      -- Propagando o erro para a camada oculta (usando matriz transposta de weightsHidden)
      dA1 = tr (weightsHidden net) #> dZ2
      
      -- Derivada da ReLU: 1 se z1 > 0, senão 0
      reluDerivative z = cmap (\x -> if x > 0 then 1.0 else 0.0) z
      dZ1 = dA1 * reluDerivative z1
      
      -- Gradientes da camada Entrada -> Oculta
      dW1 = dZ1 `outer` input
      dB1 = dZ1
      
      -- 3. Atualização (Gradiente Descendente: W_novo = W_atual - alpha * dW)
      -- A função scale multiplica a matriz/vetor pelo valor (-alpha)
  in net
    { weightsInput  = weightsInput net  + scale (-alpha) dW1
    , biasHidden    = biasHidden net    + scale (-alpha) dB1
    , weightsHidden = weightsHidden net + scale (-alpha) dW2
    , biasOutput    = biasOutput net    + scale (-alpha) dB2
    }
