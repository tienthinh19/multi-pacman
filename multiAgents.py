# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).
from pacman import GameState
from util import manhattanDistance
from game import Directions
import random, util

from game import Agent


class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        oldFood = currentGameState.getFood()
        currentPos = list(successorGameState.getPacmanPosition())
        maxDistance = -10000000

        "*** YOUR CODE HERE ***"
        distance = 0
        foodList = oldFood.asList()

        if action == 'Stop':
            return -10000000

        for state in newGhostStates:
            if state.getPosition() == tuple(currentPos) and (state.scaredTimer == 0):
                return -10000000

        for food in foodList:
            distance = -1 * (manhattanDistance(food, currentPos))

            if (distance > maxDistance):
                maxDistance = distance

        return maxDistance


def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"

        def minMaxHelper(gameState, deepness, agent):
            if agent >= gameState.getNumAgents():
                agent = 0
                deepness += 1
            if (deepness == self.depth or gameState.isWin() or gameState.isLose()):
                return self.evaluationFunction(gameState)
            elif (agent == 0):
                return maxFinder(gameState, deepness, agent)
            else:
                return minFinder(gameState, deepness, agent)

        def maxFinder(gameState, deepness, agent):
            output = ["meow", -float("inf")]
            pacActions = gameState.getLegalActions(agent)

            if not pacActions:
                return self.evaluationFunction(gameState)

            for action in pacActions:
                currState = gameState.generateSuccessor(agent, action)
                currValue = minMaxHelper(currState, deepness, agent + 1)
                if type(currValue) is list:
                    testVal = currValue[1]
                else:
                    testVal = currValue
                if testVal > output[1]:
                    output = [action, testVal]
            return output

        def minFinder(gameState, deepness, agent):
            output = ["meow", float("inf")]
            ghostActions = gameState.getLegalActions(agent)

            if not ghostActions:
                return self.evaluationFunction(gameState)

            for action in ghostActions:
                currState = gameState.generateSuccessor(agent, action)
                currValue = minMaxHelper(currState, deepness, agent + 1)
                if type(currValue) is list:
                    testVal = currValue[1]
                else:
                    testVal = currValue
                if testVal < output[1]:
                    output = [action, testVal]
            return output

        outputList = minMaxHelper(gameState, 0, 0)
        return outputList[0]

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        def expectimax(state, depth, agentIndex):
            numAgents = state.getNumAgents()
            if depth == self.depth or state.isWin() or state.isLose():
                return self.evaluationFunction(state), None

            if agentIndex == 0:  # Pacman's turn (maximizer)
                return maxValue(state, depth, agentIndex)
            else:  # Ghosts' turn (expectation)
                return expValue(state, depth, agentIndex)
        def maxValue(state, depth, agentIndex):
            v = float('-inf')
            bestAction = None
            for action in state.getLegalActions(agentIndex):
                successor = state.generateSuccessor(agentIndex, action)
                value, _ = expectimax(successor, depth, agentIndex + 1)
                if value > v:
                    v = value
                    bestAction = action
            return v, bestAction

        def expValue(state, depth, agentIndex):
            v = 0
            ghostActions = state.getLegalActions(agentIndex)
            prob = 1.0 / len(ghostActions)  # Uniform probability
            nextAgentIndex = (agentIndex + 1) % state.getNumAgents()
            nextDepth = depth + 1 if nextAgentIndex == 0 else depth

            for action in ghostActions:
                successor = state.generateSuccessor(agentIndex, action)
                value, _ = expectimax(successor, nextDepth, nextAgentIndex)
                v += value * prob
            return v, None

        _, action = expectimax(gameState, 0, 0)
        return action

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function.
    """
    # Extract useful information from the current game state
    pacmanPos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]
    capsules = currentGameState.getCapsules()
    foodList = food.asList()

    # Initial evaluation score is the current game score
    score = currentGameState.getScore()

    # Factor in the distance to the nearest food
    if foodList:
        minFoodDistance = min(manhattanDistance(pacmanPos, foodPos) for foodPos in foodList)
        score += 10.0 / minFoodDistance

    # Factor in the distance to the nearest capsule
    if capsules:
        minCapsuleDistance = min(manhattanDistance(pacmanPos, capsule) for capsule in capsules)
        score += 50.0 / minCapsuleDistance

    # Factor in the distance to the nearest ghost and scared times
    for ghostState, scaredTime in zip(ghostStates, scaredTimes):
        ghostPos = ghostState.getPosition()
        distanceToGhost = manhattanDistance(pacmanPos, ghostPos)
        
        if scaredTime > 0:
            # If the ghost is scared, closer is better (we can eat it)
            score += 200.0 / distanceToGhost
        else:
            # If the ghost is not scared, avoid it
            if distanceToGhost > 0:
                score -= 2.0 / distanceToGhost

    # Factor in the number of remaining food pellets and capsules
    score -= len(foodList) * 20  # Penalty for each remaining food pellet
    score -= len(capsules) * 100  # Penalty for each remaining capsule

    # Factor in the scared times of the ghosts
    score += sum(scaredTimes) * 10

    return score

# Abbreviation
better = betterEvaluationFunction

class AlphaBetaAgent(MultiAgentSearchAgent):

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        with alpha-beta pruning.
        """
        def alphaBeta(state, depth, agentIndex, alpha, beta):
            numAgents = state.getNumAgents()
            if depth == self.depth or state.isWin() or state.isLose():
                return self.evaluationFunction(state), None

            if agentIndex == 0:  # Pacman's turn (maximizer)
                return maxValue(state, depth, agentIndex, alpha, beta)
            else:  # Ghosts' turn (minimizer)
                return minValue(state, depth, agentIndex, alpha, beta)

        def maxValue(state, depth, agentIndex, alpha, beta):
            v = float('-inf')
            bestAction = None
            for action in state.getLegalActions(agentIndex):
                successor = state.generateSuccessor(agentIndex, action)
                value, _ = alphaBeta(successor, depth, agentIndex + 1, alpha, beta)
                if value > v:
                    v = value
                    bestAction = action
                if v > beta:
                    return v, bestAction
                alpha = max(alpha, v)
            return v, bestAction

        def minValue(state, depth, agentIndex, alpha, beta):
            v = float('inf')
            bestAction = None
            numAgents = state.getNumAgents()
            nextAgentIndex = (agentIndex + 1) % numAgents
            nextDepth = depth + 1 if nextAgentIndex == 0 else depth

            for action in state.getLegalActions(agentIndex):
                successor = state.generateSuccessor(agentIndex, action)
                value, _ = alphaBeta(successor, nextDepth, nextAgentIndex, alpha, beta)
                if value < v:
                    v = value
                    bestAction = action
                if v < alpha:
                    return v, bestAction
                beta = min(beta, v)
            return v, bestAction

        _, action = alphaBeta(gameState, 0, 0, float('-inf'), float('inf'))
        return action

