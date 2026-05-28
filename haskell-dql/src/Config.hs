module Config where

-- | Taxa de aprendizado da rede
alpha :: Double
alpha = 0.001

-- | Fator de desconto
gamma :: Double
gamma = 0.99

-- | Taxa de exploração inicial
epsilon :: Double
epsilon = 1.0

-- | Decaimento de epsilon
epsilonDecay :: Double
epsilonDecay = 0.995

-- | Epsilon mínimo
epsilonMin :: Double
epsilonMin = 0.01

-- | Tamanho do Replay Buffer
replayBufferSize :: Int
replayBufferSize = 2000

-- | Tamanho do Mini-Batch
batchSize :: Int
batchSize = 32

-- | Frequência de atualização da Target Network
targetUpdateFreq :: Int
targetUpdateFreq = 10

-- | Número de episódios de treino
numEpisodes :: Int
numEpisodes = 500
