module ReplayBuffer where

import Types (Action, Reward, StateVector)
import System.Random (StdGen, randomR)
import qualified Config

-- | O Experience Replay é necessário para estabilizar o treino da rede neural no DQL.
-- Ele armazena experiências passadas e permite treinar em mini-batches aleatórios,
-- o que quebra a correlação temporal entre amostras consecutivas e evita que a rede "esqueça"
-- experiências anteriores ao superajustar para as mais recentes.

-- | Representa uma única transição no ambiente: (Estado, Ação, Recompensa, Próximo Estado, Terminal)
type Experience = (StateVector, Action, Reward, StateVector, Bool)

-- | O Replay Buffer é simplesmente uma lista de experiências armazenadas.
-- Simplificamos para lista nativa, mas mantendo a restrição de tamanho máximo.
type ReplayBuffer = [Experience]

-- | Retorna um buffer vazio
emptyBuffer :: ReplayBuffer
emptyBuffer = []

-- | Adiciona uma nova experiência ao buffer. Se o buffer atingir o tamanho máximo,
-- remove a experiência mais antiga (que está no final da lista, já que adicionamos no início).
-- Nota: Importamos Config diretamente ao invés de passar por parâmetro.
addExperience :: ReplayBuffer -> Experience -> ReplayBuffer
addExperience buffer exp =
  let newBuffer = exp : buffer
  in take Config.replayBufferSize newBuffer

-- | Amostra aleatoriamente batchSize experiências do buffer (amostragem SEM reposição).
-- Retorna a lista de experiências amostradas e o StdGen atualizado.
sampleBatch :: ReplayBuffer -> StdGen -> ([Experience], StdGen)
sampleBatch buffer gen = sample Config.batchSize buffer gen []
  where
    sample :: Int -> ReplayBuffer -> StdGen -> [Experience] -> ([Experience], StdGen)
    sample 0 _ g acc = (acc, g)
    sample _ [] g acc = (acc, g) -- Prevenção caso o buffer seja menor que o batch (não deve ocorrer devido ao bufferReady)
    sample k b g acc =
      let maxIdx = length b - 1
          (idx, g') = randomR (0, maxIdx) g
          (xs, y:ys) = splitAt idx b
      in sample (k - 1) (xs ++ ys) g' (y : acc)

-- | Retorna True apenas quando o buffer tiver acumulado experiências suficientes
-- para extrair pelo menos um mini-batch completo.
bufferReady :: ReplayBuffer -> Bool
bufferReady buffer = length buffer >= Config.batchSize
