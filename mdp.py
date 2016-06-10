# Joey Velez-Ginorio
# MDP Implementation
# ---------------------------------
# - Includes BettingGame example


import numpy as np
import random
from abc import ABCMeta
from abc import abstractmethod

import pdb # for debugging, remove later

class MDP(object):
	""" 
		Defines an Markov Decision Process containing:
	
		- States, s 
		- Actions, a
		- Rewards, r(s,a)
		- Transition Matrix, t(s,a,_s)

		Includes a set of abstract methods for extended class will
		need to implement.

	"""

	__metaclass__ = ABCMeta
	
	def __init__(self, states=None, actions=None, rewards=None, transitions=None):
		self.s = np.array(states)
		self.a = np.array(actions)
		self.r = np.array(rewards)
		self.t = np.array(transitions)
		
		self.discount = .8

		# Value iteration will update this
		self.values = None
		self.policy = None

	@abstractmethod
	def getStartState(self, state):
		"""
			Returns the start state of MDP
		"""
		raise NotImplementedError()

	@abstractmethod
	def isTerminal(self, state):
		"""
			Checks if MDP is in terminal state.
		"""
		raise NotImplementedError()

	def getTransitionStatesAndProbs(self, state, action):
		"""
			Returns the list of transition probabilities
		"""
		return self.t[state][action][:]

	def getReward(self, state):
		"""
			Gets reward for transition from state->action->nextState.
		"""
		return self.r[state]


	def takeAction(self, state, action):
		"""
			Take an action in an MDP, return the next state

			Chooses according to probability distribution of state transitions,
			contingent on actions.
		"""
		return np.random.choice(self.s, p=self.getTransitionStatesAndProbs(state, action))	

	def valueIteration(self, epsilon=.01):
		"""
			Performs value iteration to populate the values of all states in
			the MDP. 

			Params:
				- epsilon: Determines limit of convergence
		"""

		# Initialize V_0 to zero
		self.values = np.zeros(len(self.s))

		# Loop until convergence
		while True:

			# To be used for convergence check
			oldValues = np.copy(self.values)

			for i, state in enumerate(self.s):

				# Take max over all possible actions in state
				max_a = 0

				for j, action in enumerate(self.a):

					# Account for all possible states a particular action can take you to
					sum_nextState = 0
					for k, nextState in enumerate(self.s):
						sum_nextState += self.getTransitionStatesAndProbs(i,j)[k] * \
						(self.getReward(i) + self.discount*self.values[k])

					if sum_nextState > max_a:
						max_a = sum_nextState
						

				self.values[i] = max_a

	
			print np.max(np.abs(self.values - oldValues))
			
			# Convergence Check
			if np.max(np.abs(self.values - oldValues)) <= epsilon:
				break
			
	def extractPolicy(self, tau=1000):
		"""
			Extract policy from values after value iteration runs.
		"""
		self.policy = np.zeros([len(self.s),len(self.a)])

		for i in range(len(self.s)):

			# Take max over all possible actions in state
			max_a = 0
			a_vals = list()
			state_policy = np.zeros(len(self.a))

			for j in range(len(self.a)):

				# Account for all possible states a particular action can take you to
				sum_nextState = 0
				for k in range(len(self.s)):
					sum_nextState += self.getTransitionStatesAndProbs(i,j)[k] * \
					(self.getReward(i) + self.discount*self.values[k])

				state_policy[j] = sum_nextState
			
			# Softmax the policy			
			state_policy -= np.max(state_policy)
			state_policy = np.exp(state_policy / float(tau))
			state_policy /= state_policy.sum()

			self.policy[i] = state_policy

			del a_vals[:]
				

	def extractDeterministicPolicy(self):
		"""
			Extract policy from values after value iteration runs.
		"""
		self.policy = np.zeros(len(self.s))

		for i in range(len(self.s)):

			# Take max over all possible actions in state
			max_a = 0

			for j in range(len(self.s)):

				# Account for all possible states a particular action can take you to
				sum_nextState = 0
				for k in range(len(self.s)):
					sum_nextState += self.getTransitionStatesAndProbs(i,j)[k] * \
					(self.getReward(i) + self.discount*self.values[k])

				if sum_nextState > max_a:
					max_a = sum_nextState
					self.policy[i] = j	


	def simulate(self, state):

		""" 
			Runs the solver for the MDP, conducts value iteration, extracts policy,
			then runs simulation of problem.

			NOTE: Be sure to run value iteration (solve values for states) and to
		 	extract some policy (fill in policy vector) before running simulation
		"""
		
		# Run simulation using policy until terminal condition met
		
		while not self.isTerminal(state):

			# Determine which policy to use (non-deterministic)
			policy = self.policy[np.where(self.s == state)[0][0]]
			p_policy = self.policy[np.where(self.s == state)[0][0]] / \
						self.policy[np.where(self.s == state)[0][0]].sum()

			# Get the parameters to perform one move
			stateIndex = np.where(self.s == state)[0][0]
			policyChoice = np.random.choice(policy, p=p_policy)
			actionIndex = np.random.choice(np.array(np.where(self.policy[state][:] == policyChoice)).ravel())

			# Take an action, move to next state
			nextState = self.takeAction(stateIndex, actionIndex)

			print "In state: {}, taking action: {}, moving to state: {}".format(
				state, self.a[actionIndex], nextState)

			# End game if terminal state reached
			state = int(nextState)
			if self.isTerminal(state):

				# print "Terminal state: {} has been reached. Simulation over.".format(state)
				return state



class BettingGame(MDP):

	""" 
		Defines the Betting Game:

		Problem: A gambler has the chance to make bets on the outcome of 
		a fair coin flip. If the coin is heads, the gambler wins as many
		dollars back as was staked on that particular flip - otherwise
		the money is lost. The game is won if the gambler obtains $100,
		and is lost if the gambler runs out of money (has 0$). This gambler
		did some research on MDPs and has decided to enlist them to assist
		in determination of how much money should be bet on each turn. Your 
		task is to build that MDP!

		Params: 

				startCash: Starting amount to bet with
				pHead: Probability of coin flip landing on heads
					- Use .5 for fair coin, else choose a bias [0,1]

	"""

	def __init__(self, startCash=50, pHeads=.5):

		MDP.__init__(self)
		self.cash = startCash
		self.currentState = startCash
		self.pHeads = pHeads
		self.setBettingGame(startCash, pHeads)

	def getStartState(self):
		"""
			Returns the start state of MDP
		"""
		return self.cash

	def isTerminal(self, state):
		"""
			Checks if MDP is in terminal state.
		"""
		return True if state is 100 or state is 0 else False

	def setBettingGame(self, startCash=50, pHeads=.5):

		""" 
			Initializes the MDP to the starting conditions for 
			the betting game. 

			Params:
				startCash = Amount of starting money to spend
				pHeads = Probability that coin lands on head
					- .5 for fair coin, otherwise choose bias

		"""

		# This is how much we're starting with
		self.cash = startCash
		self.pHeads = pHeads



		# Initialize all possible states
		self.s = np.arange(101)

		# Initialize possible actions
		self.a = np.arange(101)

		# Initialize rewards
		self.r = np.zeros(101)
		self.r[0] = -100
		self.r[100] = 100

		# Initialize transition matrix
		temp = np.zeros([len(self.s),len(self.a),len(self.s)])

		# List comprehension using tHelper to determine probabilities for each index
		self.t = [self.tHelper(i[0], i[1], i[2], self.pHeads) for i,x in np.ndenumerate(temp)]
		self.t = np.reshape(self.t, np.shape(temp))

	def tHelper(self, x, y, z , pHeads):

		""" 
			Helper function to be used in a list comprehension to quickly
			generate the transition matrix. Encodes the necessary conditions
			to compute the necessary probabilities.

			Params:
			x,y,z indices
			pHeads = probability coin lands on heads

		"""
	 
		# If you bet no money, you will always have original amount
		if x + y is z and y is 0:
			return 1

		# If you bet more money than you have, no chance of any outcome
		elif y > x and x is not z:
			return 0

		# If you bet more money than you have, returns same state with 1.0 prob.
		elif y > x and x is z:
			return 1

		# Chance you lose
		elif x - y is z:
			return 1 - pHeads

		# Chance you win
		elif x + y is z:
			return pHeads

		# Edge Case: Chance you win, and winnings go over 100
		elif x + y > z and z is 100:
			return pHeads

		else:
			return 0 

		return 0
 
