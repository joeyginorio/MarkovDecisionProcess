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

	def up(self, state):
		"""
			Move agent up, uses scalar position.
		"""
		return 
	
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


