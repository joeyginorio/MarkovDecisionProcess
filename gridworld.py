# Joey Velez-Ginorio
# Gridworld Implementation
# ---------------------------------
# - Includes BettingGame example

from mdp import MDP
import numpy
import random


class GridWorld(MDP):
	"""
		 Defines a gridworld environment to be solved by an MDP!
		 Uses the grid example from the Sutton, Barto text.

		 	GridWorld Map:

			A = agent
		   ||| = obstacle
		   +1/-1 = Reward
		   
			-----------------
			|	|	|	|+1	|
			-----------------
			|	||||	|-1	|	
			-----------------
			| A	|	|	|	|
			-----------------

	"""
	def __init__(self):

		MDP.__init__(self)
		self.rowLen = 4
		self.colLen = 3

	def getStartState(self, state):
		"""
			Specifies starting coordinate for gridworld.
		"""
		return 8 


	def isTerminal(self, state):
		"""
			Specifies terminal conditions for gridworld.
		"""
		return True if state is 3 or state is 7 else False

	def isObstacle(self, sCoord):
		""" 
			Checks if a state is a wall or obstacle.
		"""
		if sCoord[0] is 1 and sCoord[1] is 1:
			return True

		if sCoord[0] > (self.colLen - 1) or sCoord[0] < 0:
			return True

		if sCoord[1] > (self.rowLen - 1) or sCoord[1] < 0:
			return True

	def takeAction(self, sCoord, action):
		"""
			Receives an action value, performs associated movement.
		"""
		if action is 0:
			return self.up(sCoord)


	def up(self, sCoord):
		"""
			Move agent up, uses state coordinate.
		"""
		newCoord = np.copy(sCoord)
		newCoord[0] -= 1

		# Check if action takes you to a wall/obstacle
		if not self.isObstacle(newCoord):
			return newCoord

		# You hit a wall, return original coord
		else:
			return sCoord

	def down(self, sCoord):
		"""
			Move agent down, uses state coordinate.
		"""
		newCoord = np.copy(sCoord)
		newCoord[0] += 1

		# Check if action takes you to a wall/obstacle
		if not self.isObstacle(newCoord):
			return newCoord

		# You hit a wall, return original coord
		else:
			return sCoord

	def left(self, sCoord):
		"""
			Move agent left, uses state coordinate.
		"""
		newCoord = np.copy(sCoord)
		newCoord[1] -= 1

		# Check if action takes you to a wall/obstacle
		if not self.isObstacle(newCoord):
			return newCoord

		# You hit a wall, return original coord
		else:
			return sCoord

	def right(self, sCoord):
		"""
			Move agent right, uses state coordinate.
		"""

		newCoord = np.copy(sCoord)
		newCoord[1] += 1

		# Check if action takes you to a wall/obstacle
		if not self.isObstacle(newCoord):
			return newCoord

		# You hit a wall, return original coord
		else:
			return sCoord

	def coordToScalar(self, sCoord):
		""" 
			Convert state coordinates to corresponding scalar state value.
		"""
		return sCoord[0]*(self.rowLen) + sCoord[1]

	def scalarToCoord(self, scalar):
		"""
			Convert scalar state value into coordinates.
		"""
		return np.array([scalar / self.rowLen, scalar % self.rowLen])

	def getPossibleActions(self, sCoord):
		"""
			Will return a list of all possible actions from a current state.
		"""
		possibleActions = list()

		if self.up(sCoord) is not sCoord:
			possibleActions.append(0)

		if self.down(sCoord) is not sCoord:
			possibleActions.append(1)

		if self.left(sCoord) is not sCoord:
			possibleActions.append(2)

		if self.right(sCoord) is not sCoord:
			possibleActions.append(3)
		
		return possibleActions

	
	def setGridWorld(self):
		"""
			Initializes states, actions, rewards, transition matrix.
		"""

		# 12 Possible coordinate positions
		self.s = np.arange(12)

		# 4 Actions {Up, Down, Left, Right}
		self.a = np.arange(4)

		# 2 Reward Zones
		self.r = np.zeros(len(self.s))
		self.r[3] = 100
		self.r[7] = -100

		# Transition Matrix
		temp = np.zeros([len(self.s),len(self.a),len(self.s)])

	

