module QTable where

import Types (State(..), Action(..), Reward, QValue, QTable)
import Env (gridSize, initState, step)
import qualified Config
import qualified Data.Map.Strict as Map
import Data.List (maximumBy)
import Data.Ord (comparing)

-- | Inicializa todos os pares (State, Action) com QValue 0.0
initQTable :: QTable
initQTable = Map.fromList [ ((State r c, a), 0.0)
                          | r <- [0 .. gridSize - 1]
                          , c <- [0 .. gridSize - 1]
                          , a <- [minBound .. maxBound] ]

-- | Busca o QValue; se não existir no map, retorna 0.0 por segurança
lookupQ :: QTable -> State -> Action -> QValue
lookupQ qt s a = Map.findWithDefault 0.0 (s, a) qt

-- | Retorna a ação com o maior Q-value para o estado dado
bestAction :: QTable -> State -> Action
bestAction qt s =
  let actions = [minBound .. maxBound]
      qValues = [ (a, lookupQ qt s a) | a <- actions ]
  in fst $ maximumBy (comparing snd) qValues

-- | Atualiza a Q-Table usando a Equação de Bellman explícita.
-- Nota: Optamos por importar as variáveis de Config diretamente em vez de 
-- passá-las como parâmetro, já que Config foi criado como um módulo de constantes.
updateQTable :: QTable -> State -> Action -> Reward -> State -> QTable
updateQTable qt s a reward nextS =
  let currentQ = lookupQ qt s a
      nextMaxQ = lookupQ qt nextS (bestAction qt nextS)
      
      -- Conversão de tipos de Double (Config) para Float (QValue/Reward)
      alphaF = realToFrac Config.alpha
      gammaF = realToFrac Config.gamma
      
      -- BELLMAN EQUATION MAPPING:
      -- tdError = reward + gamma * maxNextQ - currentQ
      tdError = reward + (gammaF * nextMaxQ) - currentQ
      
      -- newQ = currentQ + alpha * tdError
      newQ = currentQ + (alphaF * tdError)
      
  in Map.insert (s, a) newQ qt

-- | Teste unitário simples para validação do ambiente (Andaime)
-- Executa 10 passos utilizando a Q-Table e a função bestAction.
testEnvironment :: IO ()
testEnvironment = do
  putStrLn "Iniciando teste de validação do GridWorld (10 steps)..."
  let loop 0 _ _ accReward = putStrLn $ "\nTeste finalizado. Recompensa acumulada: " ++ show accReward
      loop n qt s accReward = do
        let a = bestAction qt s
        let (nextS, r, isTerm) = step s a
        let qt' = updateQTable qt s a r nextS
        
        putStrLn $ "Passo " ++ show (11 - n) ++ ": Estado " ++ show s ++ " | Ação " ++ show a ++ " | Rec. " ++ show r ++ " | Q(s,a) = " ++ show (lookupQ qt' s a)
        
        if isTerm
           then putStrLn $ "\nObjetivo alcançado! Recompensa acumulada final: " ++ show (accReward + r)
           else loop (n - 1) qt' nextS (accReward + r)
           
  loop 10 initQTable initState 0.0
