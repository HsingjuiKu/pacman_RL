# mdpAgents.py
# parsons/20-nov-2017
#
# Version 1
#
# The starting point for CW2.
#
# Intended to work with the PacMan AI projects from:
#
# http://ai.berkeley.edu/
#
# These use a simple API that allow us to control Pacman's interaction with
# the environment adding a layer on top of the AI Berkeley code.
#
# As required by the licensing agreement for the PacMan AI we have:
#
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

# The agent here is was written by Simon Parsons, based on the code in
# pacmanAgents.py

from pacman import Directions
from game import Agent
import api
import collections
import util
"""
Author: Xingrui Gu
I'm glad to have the opportunity to share my work ideas with you. I created a node class for each node in the whole game world. In the node class, 
there are an X and Y variable to record the coordinate position, as well as the utility and reward variables of each node. Then I will create the 
gridworld class through the node class. In this class, I will let them have corner, wall, food, ghosts and Pacman. All theoretical knowledge about 
Markov decision analysis is in MDP class. It has been founded by Bellman Equation, The Bellman Equation has three main parts. The first is
the Reward Part, and discountFactor and the max Expect Utility, so it has been constructed by functions of Value Interaction Bellman Update and 
Expect Utility.
In the last class, there are more algorithm in here, they will help Pacman run away when Ghost is too close, It will dynamics change the
relevant Node reward to force the pacman run away ghosts. I mainly Compute the adjacent node with Ghost and create a warning area. 
Then I also use the Bread First Search to find the shortes path with closest food. And In small Grid because the map is so small, So I also
use the BFS search the path between pacman and Ghost to change those node reward mark. And I also use find farest food with ghost to keep the 
distance between ghost and Pacman. More information will be shown after.

I have get some inspiritation and reference the good code and algorithm form below URL(Some Eng and some are Chinese):
Inspriation:
https://www.cxyzjd.com/article/u013010889/81909633 Author:Unknown
https://prateek-mishra.medium.com/markovian-pac-man-8dd212c5a35c Author:Prateek Mishra
https://leyankoh.com/2017/12/14/an-mdp-solver-for-pacman-to-navigate-a-nondeterministic-environment/ Author:KYLIE
https://stackoverflow.com/questions/47896461/get-shortest-path-to-a-cell-in-a-2d-array-in-python Author:Unknown
References:
I only direct reference Jason_Liu code in the MDPAgent to set the GhostZone Effect Area
https://download.csdn.net/download/j_ason_liu/10466221 Author: Jason_liu
"""


#Initial the Reward Mark
REWARD_FOOD = 10
REWARD_OPEN = -1
REWARD_GHOST = -2000
#From the API get the direction probability
PROBABILITY_SUCCESS = api.directionProb
PROBABILITY_FAIL = (1-PROBABILITY_SUCCESS)/2
#Set the direction vector
directVector = {'North':(0,1), 'South':(0,-1),'East':(1,0),'West':(-1,0)}
#Set the Reward Mechanism
rewardMechanism = {'o': REWARD_OPEN,'f':REWARD_FOOD, 'g':REWARD_GHOST, 'w':0}
gamma = 0.8
coverCoefficient = 0.1
ActionList = ['North','South','East','West']

class Node():

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.status = None
        self.utility = None
        self.reward = 0

    def setStatus(self, status):
        """
        Setter Method, For setting every node status, identity every node element
        Args:
            status: the status of wall, ghost, food and opennode

        """
        self.status = status

    def setUtility(self, utility):
        """
        Setter Method, For setting the utility for the node
        Args:
            utility: Utility in Bellman Equation

        """
        self.utility = utility
    def setReward(self, reward):
        self.reward = reward

    def getLocation(self):
        """
        Get Method to get the tuple of x, y coordinates
        Returns:
            tuple of (x,y)
        """
        return (self.x, self.y)


class GridWorld():
    def __init__(self, state):
        self.corners = api.corners(state)
        self.walls = api.walls(state)
        self.food = api.food(state)
        self.ghosts = api.ghosts(state)
        self.pacMan = api.whereAmI(state)
        self.boundaries = None
        self.world = self.buildWorld()
        self.buildStatus()

    def getNode(self, location, nodes):
        """
        From the world node get the (x,y) node object
        Args:
            location: (x,y) coordinate
            nodes: node object

        Returns:
            Node object

        """
        for n in nodes:
            if n.getLocation() == location:
                return n

    def setStatus(self, nodes):
        """
        Initial the node status by set Method 'o' for open node
        'f' for food node 'g' for ghost node 'w' for wall node
        Args:
            nodes: List of all node
        """
        openNode = []
        for n in self.world:
            if n.getLocation() not in (self.food + self.walls + self.ghosts):
                openNode.append(n)
        status = {'o': openNode, 'f': self.food, 'g': self.ghosts, 'w': self.walls}
        for n in nodes:
            for key, value in status.items():
                if n.getLocation() in value:
                    n.status = key

    def buildStatus(self):
        """
        Call setStatus Method

        """
        self.setStatus(self.world)

    def buildWorld(self):
        """
        Return a list of node object that make up the nodes of the map
        Returns:
            nodes: List of node

        """
        for point in self.corners:
            if point[0] and point[1] != 0:
                boundaryPoint = point
        Height = boundaryPoint[0]
        Width = boundaryPoint[1]
        if Height < 10:
            self.boundaries = [(0, Height), (0, Width)]
        else:
            self.boundaries = self.adjustmentMap()
        world = []
        for y in range(int(self.boundaries[1][0]), int(self.boundaries[1][1] + 1)):
            for x in range(int(self.boundaries[0][0]), int(self.boundaries[0][1] + 1)):
                world.append(self.createNode(x,y))
        return world

    def createNode(self,x,y):
        """
        Build the node class function
        Args:
            x: the coordinate of x
            y: the coordinate of y

        Returns:
            Node(x,y): the coordinate of (x,y) node class include the reward, utility and status
        """
        return Node(x,y)

    def adjustmentMap(self):
        """
        Only create the world by using the valid zone for ghost and pacman, if every iteraction both need to create huge world,
        it will lose too much time
        Returns: a new boundary list by tuple of coordinate

        """
        usefulPoint = self.ghosts + self.food + [self.pacMan]
        #Find the boundary of current non-o node
        maxY = max([u[1] for u in usefulPoint])
        minY = min([u[1] for u in usefulPoint])
        maxX = max([u[0] for u in usefulPoint])
        minX = min([u[0] for u in usefulPoint])
        newBoundaries = [(minX, maxX), (minY, maxY)]
        return newBoundaries



class MDP():
    def __init__(self, currState, nodes):
        self.nodes = nodes
        self.currState = currState
        self.transitionModel = {'Success': PROBABILITY_SUCCESS, 'Failure': (1-PROBABILITY_SUCCESS)/2}
        self.discountFactor = gamma
        self.conver = coverCoefficient

    def valueIteration(self):
        """
        Use the value iteration function to determing utilities of every state. The number of iteration
        has been determined by conver constant. In this method will call the bellman equation
        """
        for n in self.nodes:
            n.utility = 0

        condition = False
        while condition is False:
            utilities = []
            for n in self.nodes:
                utilities.append ((n, self.BellmanEq(n)))
            for u in utilities:
                difference = abs(u[0].utility-u[1])
                if difference < self.conver:
                    condition = True
            for u in utilities:
                u[0].utility = u[1]

    def BellmanEq(self, state):
        """
        Calculate the bellman equation of MDP to get each utility,
        the bellman function is U = R + Y*E
        Args:
            state: node containing each states status, reward and utility

        Returns:
            utility: the value of utility

        """
        utility = state.reward + (self.discountFactor * self.MEU(state, directVector.keys())[1])
        return utility

    def MEU(self, node, actions):
        """
        Prepare to compute the Bellman Equation MEU part by getMEU function
        Args:
            node: the node
            actions: action list

        Returns:
            meu: tuple (node, meu)
        """
        Utility = {}
        for move in actions:
            Vec = directVector[move]
            targetLoc = tuple(map(sum, zip(node.getLocation(), Vec)))
            targetNode = next((n for n in self.nodes if n.getLocation() == targetLoc), None)
            targetNode = node if targetNode == None else targetNode

            adjNode = self.getAdjState(Vec, node)
            moveExpUtil = sum([self.transitionModel['Failure'] * s.utility for s in adjNode])+self.transitionModel['Success'] * targetNode.utility

            Utility[move] = moveExpUtil
        meu = max(Utility.values())

        return next(((k, v) for k, v in Utility.items() if v == meu), None)

    def getAdjState(self, dirVec, state):
        """
        Determines the adjacent node of the
        Args:
            dirVec:tuple contain direction vector
            state: current state object

        Returns:
            adjStates: list of adjacent states

        """
        adjAction =[(0, 1), (0, -1)] if dirVec[0] == 0 else [(1, 0), (-1, 0)]
        adjLoc = [tuple(map(sum, zip(d, state.getLocation()))) for d in adjAction]
        adjStates = [s for loc in adjLoc for s in self.nodes if s.getLocation() == loc]
        return adjStates

class MDPAgent(Agent):
    # Constructor: this gets run when we first invoke pacman.py
    def __init__(self):
        #print "Starting up MDPAgent!"
        name = "Pacman"
        self.world = None


    # Gets run after an MDPAgent object is created and once there is
    # game state to access.
    def registerInitialState(self, state):
        return
        #print "Running registerInitialState for MDPAgent!"
        #print "I'm at:"
        #print api.whereAmI(state)


    # This is what gets run in between multiple games
    def final(self, state):
        # print "Looks like the game just ended!"
        return

    # For now I just move randomly
    def getAction(self, state):
        currLoc = api.whereAmI(state)
        self.world = GridWorld(state)
        legalActions = api.legalActions(state)
        if Directions.STOP in legalActions:
            legalActions.remove(Directions.STOP)

        pathGhost = []
        foodDistanceList = {}
        Size_of_map =  self.world.boundaries[0][1]
        if Size_of_map < 10: #Small Grid
            for i in self.world.ghosts:
                pathGhost.append((self.BFSfindpath(currLoc, i)))
            for i in self.world.food:
                foodDistanceList[i] = self.BFSfindpath(currLoc, i)
            shortPath = foodDistanceList.get(min(foodDistanceList.keys()))
            if shortPath is not None:
                if len(shortPath) > 0:
                    for i in shortPath:
                        self.findClosestfood(i)
                for i in pathGhost:
                    if i is None:
                        continue
                    if len(i) < 10:
                        self.setWarningArea(i)
        else: #Medium Classic
            g1 = self.world.ghosts[0]
            g2 = self.world.ghosts[1]
            path1 = self.BFSfindpath(currLoc,g1)
            path2 = self.BFSfindpath(currLoc,g2)
            if g1 > g2:
                distanceGhost = g2
            else:
                distanceGhost = g1
            if distanceGhost > 4:
                if distanceGhost < 10:
                    farfood = self.checkThefarfood()
                    path = self.BFSfindpath(currLoc,farfood)
                    if path != None:
                        for i in path:
                            self.findClosestfood(i)
                else:
                    foodDistanceList = {}
                    if len(self.world.food) > 3:
                        checkList = self.safeFood()
                    else:
                        checkList = self.world.food
                    for f in checkList:
                        path = self.BFSfindpath(currLoc, f)
                        if self.checkPathGhost(path) is True:
                            foodDistanceList[f] = path
                    short = 1000
                    short_path = []
                    for i in foodDistanceList.values():
                        if i is None:
                            continue
                        if short > len(i):
                            short = len(i)
                            short_path = i
                    if currLoc in short_path:
                        short_path.remove(currLoc)
                    if len(short_path) > 0:
                        for i in short_path:
                            self.findClosestfood(i)
            if path1 is not None and len(path1)<6:
                self.setWarningArea(path1)
            if path2 is not None and len(path2)<6:
                self.setWarningArea(path2)
            self.buildGhosts(self.world.world,state)

        #Initial the world Node
        for n in self.world.world:
            if n.getLocation() in self.world.ghosts:
                n.status = 'g'
            if n.status != 's' and n.status is not None:
                n.setReward(rewardMechanism[n.status])
        states = [n for n in self.world.world if n.status != 'w']
        currState = next((n for n in states if n.getLocation() == currLoc), None)
        #Call The MDP computing
        mdp = MDP(currState, states)
        mdp.valueIteration()
        meuMove = mdp.MEU(mdp.currState, legalActions)[0]
        return api.makeMove(meuMove, legalActions)

    def isWall(self, loc):
        """
        Check the loc is or not wall node
        Args:
            loc: (x,y)

        Returns:
            Boolean Value True or False
        """
        return loc in self.world.walls

    def inRegion(self, loc):
        """
        Check whether the node is in the world
        Args:
            loc: (x,y)

        Returns:
             Boolean Value True or False
        """
        for n in self.world.world:
            if loc == n.getLocation():
                return True
        return False
    def checkThefarfood(self):
        """
        Find the Farest food with ghost from the food list
        Returns:
            farfood: coordinate of food
        """
        distance = {}
        for f in self.world.food:
            g1 = self.world.ghosts[0]
            g2 = self.world.ghosts[0]#I have change it from 0 to 1 Before I sumbited it is 0 I am forgetting it
            if util.manhattanDistance(f,g1) > util.manhattanDistance(f,g2):
                distance[f] = util.manhattanDistance(f,g1)
            else:
                distance[f] = util.manhattanDistance(f,g2)
        newd = {v:k for k, v in distance.items()}
        farfood = newd.get(max(distance.values()))
        return farfood
    def BFSfindpath(self, start, destination):
        """
        By the BFS search algorithm to find the shortest path in the world
        Args:
            start: start point
            destination: destination point

        Returns:
            path: The list of shorttest Path
        """
        queue = collections.deque([[start]])
        visited = set([start])
        while queue:
           path = queue.popleft()
           curr_x,curr_y = path[-1]
           if curr_x == destination[0] and curr_y == destination[1]:
               return path
           for x2, y2 in ((curr_x+1,curr_y), (curr_x-1,curr_y), (curr_x,curr_y+1), (curr_x,curr_y-1)):
               newLoc = (x2, y2)
               if self.isWall(newLoc) is False and newLoc not in visited and self.inRegion(newLoc):
                   queue.append(path+[newLoc])
                   visited.add(newLoc)


    def setWarningArea(self, path):
        """
        Set the dangerous area of the path point
        Args:
            path: list of node coordinate

        """
        node = self.world.world
        for i in path:
            for n in node:
                if n.getLocation() == i:
                    n.status = 's'
                    n.reward = -600

    def findClosestfood(self, adjNode):
        """
        Setting
        Args:
            adjNode:

        Returns:

        """
        for n in self.world.world:
            if adjNode == n.getLocation():
                n.status = 's'
                n.reward = 200

    def safeFood(self):
        """
        This Method use to find all safe food(mean around it no Ghosts
        Returns:

        """
        safe = []
        for f in self.world.food:
            if len(self.world.ghosts) == 2:
                g1 = self.world.ghosts[0]
                g2 = self.world.ghosts[1]
                if util.manhattanDistance(f,g1)>9 and util.manhattanDistance(f,g2)>9:
                    safe.append(f)
            else:
                g = self.world.ghosts[0]
                if util.manhattanDistance(f,g)>7:
                    safe.append(f)

        return safe

    def checkPathGhost(self,path):
        """
        Check whether the path point has ghost
        Args:
            path: the list of path
        Returns:
            True or False
        """
        for g in self.world.ghosts:
            if path is None:
                return False
            if g in path:
                return False
        return True

    """
    This function code reference from URL, I think that method is good!! The URL has wrote in the before.
    """
    def buildGhosts(self, nodes, state):
        """
        Build the dangerous zone of goast
        Args:
            nodes: The self.world.world, all the node in Pacman world
        """

        gEffectDict = {}

        ghostView = 2

        for ghost in api.ghostStates(state):
            if ghost[1] == 0:
                gEffectDict = self.setGhostEffect(ghost[0], gEffectDict, ghostView)

        for ghost in self.world.ghosts:
            if ghost in gEffectDict:
                gEffectDict.pop(ghost)

        for k, v in gEffectDict.items():
            n = self.world.getNode(k, nodes)
            if n is not None:
                n.reward = v
                n.status = 's'

    def setGhostEffect(self, ghost, gEffectDict, ghostView):
        """
        Builds the ghost safety zone for pacman and sets corresponding reward
        values for states based on number of steps from the ghost.
        Args:
            ghost:Coordinate of Ghost
            gEffectDict:Dictionary of ghost Effect area
            ghostView: int value ghost effect radius

        Returns:
            gEffectDict:Dictionary of ghost Effect area

        """

        distancePenalty = {1: -400, 2:-250, 3:-200, 4:-150}
        ghostArea = self.adjGhostArea(ghost, ghostView)
        if ghost in ghostArea:
            ghostArea.remove(ghost)

        for g in ghostArea:

            distance = abs(g[0] - ghost[0]) + abs(g[1]-ghost[1])
            if distance not in distancePenalty:
                continue

            if g not in gEffectDict:
                gEffectDict[g] = distancePenalty[distance]
            else:
                gEffectDict[g] += distancePenalty[distance]

        return gEffectDict
    def adjGhostArea(self, ghostLoc, ghostView):
        """
        builds ghost area based on self.ghostRadius and returns List
        if tuples with locations in the area.
        Args:
            ghostLoc:coordinate of ghost location
            ghostView:int value ghost view radius

        Returns:
            ghostArea: list of the coordinate(x,y)

        """
        mapping = [(x,y) for x in range(ghostView + 1) for y in range(ghostView + 1)]
        function = lambda z: ((z[0],z[1]), (-z[0],z[1]), (z[0], -z[1]), (-z[0], -z[1]))
        mapping = list(set([n for loc in map(function, mapping) for n in loc]))
        ghostLocLst = [ghostLoc for i in range(len(mapping))]
        ghostArea = list(map(lambda x, y: (max(x[0]+y[0],0), max(x[1]+y[1],0)), mapping, ghostLocLst))
        ghostArea = list(filter(lambda x: x not in self.world.walls, ghostArea))
        ghostArea = [g for g in ghostArea if g[0] < self.world.boundaries[0][1] and g[1] < self.world.boundaries[1][1]]
        ghostArea = [g for g in ghostArea if g[0] > self.world.boundaries[0][0] and g[1] > self.world.boundaries[1][0]]

        return ghostArea



