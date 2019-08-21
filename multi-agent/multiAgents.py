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
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

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

        "*** YOUR CODE HERE ***"
        #i will choose my features to be as follows: remaining food pallets, distance to closest ghost, and distance to closest 
        #food palet 
        #first find the values

        distToGhost = [manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates]
        distTofood =  [manhattanDistance(newPos, food) for food in newFood.asList()] 

        #assign weights to food and ghost features 

        wGhost = -10
        wFood = 10

        #define features 

        remainingFood = -len(newFood.asList())
        ghostDist = 0
        foodDist = 0 
        if len(distTofood):
            foodDist = wFood / (min(distTofood) + 1)
        if ghostDist > 0:
            ghostDist = wGhost / (min(distToGhost) + 1)

        evalFunct = successorGameState.getScore() + remainingFood + ghostDist + foodDist

        return evalFunct



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

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
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

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        val, action = self.value(state  = gameState, agent = 0, depth = self.depth)
        return action




    def value(self, state, agent, depth): 

        agents = state.getNumAgents()

        #if its temrinal state return its value and corresponding action

        if depth == 0 or state.isLose() or state.isWin(): 
            return self.evaluationFunction(state), Directions.STOP

        #if state is pacman controlled maximize value of successors 

        elif agent == 0:
            return self.maxmize(state, agent, depth)

        #if agent is opponent controlled; minimize the state 
        else:

            return self.minimize(state, agent, depth )

    def maxmize(self, state, agent, depth): 
        """assigns value and corresponding action to maximizer states """

        posActions = state.getLegalActions(agent)
        successors = [state.generateSuccessor(agent, action) for action in posActions]
        
        #check if the agent is the last acting one 
        if agent == state.getNumAgents() - 1: 
            newAgent = 0 
            newDepth = depth - 1
        #otherwise can just update
        else:
            newAgent = agent + 1 
            newDepth = depth 


        v = float('-inf')
        action = Directions.STOP 


        for posAction in posActions: 
            succesorOfAction = state.generateSuccessor(agent, posAction)
            successorVal = self.value(succesorOfAction, newAgent, newDepth)
            
            print(posAction)
            if float(successorVal[0]) > v: 
                v = float(successorVal[0]) 
                action = posAction

        return v, action

    def minimize(self, state, agent, depth): 

        posActions = state.getLegalActions(agent)
        successors = [state.generateSuccessor(agent, action) for action in posActions]

        if agent == state.getNumAgents() - 1: 
            newAgent = 0 
            newDepth = depth - 1
        else:
            newAgent = agent + 1 
            newDepth = depth 


        v = float('inf')
        action = Directions.STOP

        

        for posAction in posActions: 
            succesorOfAction = state.generateSuccessor(agent, posAction)
            successorVal = self.value(succesorOfAction, newAgent, newDepth)
            if float(successorVal[0]) < v: 
                v = float(successorVal[0])
                action = posAction
        return v, posAction
            



#make a class which contains a state's value and  action (last hope at debugging)

class stateProperties:
    def __init__(self, value, action):

        self.value = value 
        self.action = action 





class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        properties = self.prunedMinimax(state = gameState, depth = self.depth, agent = 0)
        return properties.action

    def prunedMinimax(self, state, depth, agent, alpha = float('-inf'), beta = float('inf')): 
        #if terminal state return the state's vlaues and actions

        if depth == 0 or state.isWin() or state.isLose(): 
            return stateProperties(self.evaluationFunction(state), Directions.STOP)

        #if state is pacman controlled then maximise 

        elif agent == 0: 
            return self.prunedMax(state, depth, agent, alpha, beta)

        #if state is opponent controlled then minimize 

        else: 
            return self.prunedMin(state, depth, agent, alpha, beta)


    def prunedMax(self, state, depth, agent, alpha, beta): 
        posActions = state.getLegalActions(agent)
        
        
        #check if the agent is the last acting one 
        if agent == state.getNumAgents() - 1: 
            newAgent = 0 
            newDepth = depth - 1
        #otherwise can just update
        else:
            newAgent = agent + 1 
            newDepth = depth 


        v = float('-inf')
        action = Directions.STOP 


        for posAction in state.getLegalActions(agent): 
            successorOfAction = state.generateSuccessor(agent, posAction)
            #as I try to find the values I need to account for pruning so ill run minimax 
            successorProperties = self.prunedMinimax(successorOfAction, newDepth, newAgent, alpha, beta)

            

            
            #need to see if its better than my current vlaue 


            if successorProperties.value > v: 
                v = successorProperties.value
                action = posAction

            #check if better than my MIN agents best estimate 

            if successorProperties.value > beta: 
                return successorProperties

            alpha = max(alpha, successorProperties.value)
        return stateProperties(v, action)

    def prunedMin(self, state, depth, agent, alpha, beta): 

        posActions = state.getLegalActions(agent)
        
        
        #check if the agent is the last acting one 
        if agent == state.getNumAgents() - 1: 
            newAgent = 0 
            newDepth = depth - 1
        #otherwise can just update
        else:
            newAgent = agent + 1 
            newDepth = depth 


        v = float('inf')
        action = Directions.STOP 


        for posAction in state.getLegalActions(agent): 
            successorOfAction = state.generateSuccessor(agent, posAction)
            #as I try to find the values I need to account for pruning so ill run minimax 
            successorProperties = self.prunedMinimax(successorOfAction, newDepth, newAgent, alpha, beta)
            
            #need to see if its better than my current vlaue 

            if successorProperties.value < v: 
                v = successorProperties.value 
                action = posAction

            #check if better than my MAX agents best estimate 

            if successorProperties.value < alpha: 
                return successorProperties

            beta = min(beta, successorProperties.value)
        return stateProperties(v, action)









class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"

        return self.expectimax(1,0,gameState)


    def expectimax(self, depth, agent, gameState):

        utility = []
        legalActions=gameState.getLegalActions(agent)

        if depth>self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        if Directions.STOP in legalActions:
            legalActions.remove(Directions.STOP)

        for action in legalActions:
            successor=gameState.generateSuccessor(agent, action)

            if((agent+1)>=gameState.getNumAgents()):
                utility.append(self.expectimax(depth+1, 0, successor))
                
            else:
                utility.append(self.expectimax(depth, agent+1, successor))

        if agent == 0:
            if depth == 1: 
                for i in range(len(utility)):
                    if utility[i] == max(utility):
                        return legalActions[i]
            else:
                utilV = max(utility)

        elif agent > 0: 
            utilV = float(sum(utility)/len(utility))

        return utilV


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: the better evaluation function will be a sum of weighted features which consist of distance to the nearest ghost
    accounting for whether a ghost is scared or not, remaining food pallets, and the  distance to the nearest food. 
    """
    "*** YOUR CODE HERE ***"

    
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    
    #assign weights 

    wRemainingFood = -5
    wDistanceToFood = 10 

    #start my evaluation fucntion

    evaluationFunction = currentGameState.getScore()

    #define the distance to the nearest food feature

    distTofood = [manhattanDistance(newPos, food) for food in newFood.asList()]

    foodDistFeature = 0 
    if len(distTofood):
        foodDistFeature = wDistanceToFood / (min(distTofood) + 1)

    #define the remaining food pallet feature 

    remainingFood = len(newFood.asList())
    remainingFoodFeature = wRemainingFood * remainingFood

    #define distance to the nearest ghost

    ghostDistFeature = 0 
    

    for ghost in newGhostStates: 
        
        distToGhost = manhattanDistance(newPos, ghost.getPosition())

        if distToGhost > 0: 
            if ghost.scaredTimer > 0: 
            #scared then i should chase
                wGhost = 10
                ghostDistFeature = wGhost * 1/(distToGhost + 1)

            else: 
                wGhost = -10 
                ghostDistFeature = wGhost * 1/(distToGhost + 1)



    evaluationFunction = currentGameState.getScore() + foodDistFeature + ghostDistFeature + remainingFood 
    

    return evaluationFunction








        





    


    



    

# Abbreviation
better = betterEvaluationFunction
