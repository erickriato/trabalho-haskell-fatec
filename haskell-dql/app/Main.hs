module Main (main) where

import Control.Monad.State
import System.Random (mkStdGen, StdGen)
import Control.Monad (when, forM_)

import qualified Config
import Types (Action(..), StateVector)
import qualified Types -- Para evitar conflito entre Types.State e Control.Monad.State
import Network (Network, initNetwork)
import Env (initState, step, encodeState)
import ReplayBuffer (ReplayBuffer, emptyBuffer, addExperience, bufferReady, sampleBatch)
import Agent (selectAction, decayEpsilon, trainOnBatch)

-- | ARQUITETURA DEEP Q-LEARNING (DQL)
-- Este motor acadêmico substitui completamente a Q-Table da Etapa 3 por uma
-- Rede Neural com ativação ReLU implementada via hmatrix. Em vez de indexar cada
-- estado de forma rígida, a rede mapeia observações contínuas e generaliza Q-values.
-- Para estabilizar o aprendizado via gradiente descendente, incorporamos:
-- 1. Experience Replay: Armazena experiências e amostra mini-batches aleatórios,
--    quebrando a correlação temporal e estabilizando a distribuição das amostras.
-- 2. Target Network: Uma cópia congelada da rede principal usada temporariamente
--    para calcular os alvos de Bellman de maneira estável (Double DQN simplificado).

-- Estrutura de Estado para a Mônada State
data TrainState = TrainState
  { tsMainNet   :: Network
  , tsTargetNet :: Network
  , tsBuffer    :: ReplayBuffer
  , tsEpsilon   :: Double
  , tsRng       :: StdGen
  }

-- Nosso Monad Transformer para gerenciar estados puros com side-effects IO
type DQL a = StateT TrainState IO a

-- | Função que executa um episódio completo do treinamento
runEpisode :: Int -> DQL ()
runEpisode epNum = do
  -- Inicia o loop do episódio a partir do estado inicial do ambiente
  loopEpisode Env.initState 0.0

  -- Ao final do episódio, decai a taxa de exploração (Epsilon)
  ts <- get
  let newEpsilon = decayEpsilon (tsEpsilon ts)
  put ts { tsEpsilon = newEpsilon }
  
  -- A cada 'targetUpdateFreq' episódios, sincroniza a Target Network com a Main Network
  when (epNum `mod` Config.targetUpdateFreq == 0) $ do
    liftIO $ putStrLn $ ">>> [Sincronização] Copiando pesos para a Target Network (Episódio " ++ show epNum ++ ")"
    ts' <- get
    put ts' { tsTargetNet = tsMainNet ts' }

  where
    -- Loop interno que simula os passos até atingir o estado terminal
    loopEpisode :: Types.State -> Double -> DQL ()
    loopEpisode currentState accReward = do
      ts <- get
      let encState = encodeState currentState
      
      -- 1. Política Epsilon-Greedy com a Rede Neural Principal
      let (action, newRng) = selectAction (tsMainNet ts) encState (tsEpsilon ts) (tsRng ts)
      
      -- 2. Interação com o Ambiente
      let (nextState, reward, isTerm) = step currentState action
      let encNextState = encodeState nextState
      
      -- 3. Armazena no Experience Replay
      let expTuple = (encState, action, reward, encNextState, isTerm)
      let newBuffer = addExperience (tsBuffer ts) expTuple
      
      -- Atualiza temporariamente RNG e Buffer no State
      put ts { tsBuffer = newBuffer, tsRng = newRng }
      
      -- 4. Otimização da Rede (quando o buffer tiver amostras suficientes)
      when (bufferReady newBuffer) $ do
        ts' <- get
        -- Amostra um mini-batch aleatório puro
        let (batch, rngAfterSample) = sampleBatch (tsBuffer ts') (tsRng ts')
        
        -- Backpropagation: Calcula os alvos com TargetNet, mas atualiza os pesos da MainNet
        let updatedMainNet = trainOnBatch (tsMainNet ts') (tsTargetNet ts') batch
        
        put ts' { tsMainNet = updatedMainNet, tsRng = rngAfterSample }
      
      -- Fim da transição ou continua
      if isTerm
         then do
           tsFinal <- get
           liftIO $ putStrLn $ "Episódio " ++ show epNum 
                            ++ " | Recompensa Acumulada: " ++ show (accReward + realToFrac reward) 
                            ++ " | Epsilon Atual: " ++ show (tsEpsilon tsFinal)
         else
           loopEpisode nextState (accReward + realToFrac reward)

main :: IO ()
main = do
  putStrLn "================================================================"
  putStrLn "  INICIANDO MOTOR DEEP Q-LEARNING (HaskellRL VibeCoder)         "
  putStrLn "================================================================"
  
  -- Semente determinística para reprodutibilidade
  let initialGen = mkStdGen 42
  
  -- Inicializa pesos e biases da rede principal com distribuição aleatória uniforme
  let (net, gen2) = initNetwork initialGen
  
  -- Configura a fundação inicial do treinamento
  let startState = TrainState
        { tsMainNet   = net
        , tsTargetNet = net          -- Cópia idêntica inicial congelada
        , tsBuffer    = emptyBuffer
        , tsEpsilon   = Config.epsilon
        , tsRng       = gen2
        }
        
  -- Executa o treinamento DQL por N episódios
  _ <- execStateT (forM_ [1 .. Config.numEpisodes] runEpisode) startState
  putStrLn "\nTreinamento Finalizado!"
