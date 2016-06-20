# Joey Velez-Ginorio
# Gridworld Implementation
# ---------------------------------
# - Includes BettingGame example

from mdp import MDP
from scipy.stats import uniform
from scipy.stats import beta
from scipy.stats import expon
import numpy as np
import random
import pyprind
import matplotlib.pyplot as plt



class GridWorld(MDP):
	"""
		 Defines a gridworld environment to be solved by an MDP!
	"""
	def __init__(self, goalA, goalB):

		MDP.__init__(self)
		self.rowLen = 3
		self.colLen = 3

		self.goalA = goalA
		self.goalB = goalB
		self.setGridWorld()
		self.valueIteration()
		self.extractPolicy(.01)


	def getStartState(self):
		"""
			Specifies starting coordinate for gridworld.
		"""
		return 8


	def isTerminal(self, state):
		"""
			Specifies terminal conditions for gridworld.
		"""
		return True if state == 0 or state == 6 else False

	def isObstacle(self, sCoord):
		""" 
			Checks if a state is a wall or obstacle.
		"""
		if sCoord[0] > (self.colLen - 1) or sCoord[0] < 0:
			return True

		if sCoord[1] > (self.rowLen - 1) or sCoord[1] < 0:
			return True

		return False

	def takeAction(self, sCoord, action):
		"""
			Receives an action value, performs associated movement.
		"""
		if action is 0:
			return self.up(sCoord)

		if action is 1:
			return self.down(sCoord)

		if action is 2:
			return self.left(sCoord)

		if action is 3:
			return self.right(sCoord)


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

		# 9 Possible coordinate positions + Death State
		self.s = np.arange(10)

		# 4 Actions {Up, Down, Left, Right}
		self.a = np.arange(4)

		# 2 Reward Zones
		self.r = np.zeros(len(self.s))
		self.r[0] = self.goalA
		self.r[6] = self.goalB

		# Transition Matrix
		self.t = np.zeros([len(self.s),len(self.a),len(self.s)])

		for state in range(len(self.s)):
			possibleActions = self.getPossibleActions(self.scalarToCoord(state))

			if state == 0 or state == 6:

				for i in range(len(self.a)):
					self.t[state][i][9] = 1.0

				continue
			
			for action in self.a:

				# Up
				if action == 0:

					currentState = self.scalarToCoord(state)

					nextState = self.takeAction(currentState, 0)
					self.t[state][action][self.coordToScalar(nextState)] += 1.0

					# nextState = self.takeAction(currentState, 2)
					# self.t[state][action][self.coordToScalar(nextState)] += .1

					# nextState = self.takeAction(currentState, 3)
					# self.t[state][action][self.coordToScalar(nextState)] += .1

				if action == 1:

					currentState = self.scalarToCoord(state)

					nextState = self.takeAction(currentState, 1)
					self.t[state][action][self.coordToScalar(nextState)] = 1.0

				if action == 2:

					currentState = self.scalarToCoord(state)

					nextState = self.takeAction(currentState, 2)
					self.t[state][action][self.coordToScalar(nextState)] = 1.0

				if action == 3:

					currentState = self.scalarToCoord(state)

					nextState = self.takeAction(currentState, 3)
					self.t[state][action][self.coordToScalar(nextState)] = 1.0


	def simulate(self, state):

		""" 
			Runs the solver for the MDP, conducts value iteration, extracts policy,
			then runs simulation of problem.

			NOTE: Be sure to run value iteration (solve values for states) and to
		 	extract some policy (fill in policy vector) before running simulation
		"""
		
		# Run simulation using policy until terminal condition met
		
		actions = ['up', 'down', 'left', 'right']

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
			nextState = self.takeAction(self.scalarToCoord(int(stateIndex)), int(actionIndex))
			nextState = self.coordToScalar(nextState)

			print "In state: {}, taking action: {}, moving to state: {}".format(
				self.scalarToCoord(state), actions[actionIndex], self.scalarToCoord(nextState))

			# End game if terminal state reached
			state = int(nextState)
			if self.isTerminal(state):

				print "Terminal state: {} has been reached. Simulation over.".format(self.scalarToCoord(state))
				

class InferenceMachine():
	"""
		Conducts inference via MDPs for the BettingGame.
	"""
	def __init__(self, space):
		self.sims = list()

		self.likelihood = None
		self.posterior = None
		self.prior = None

		self.space = space
		self.test = list()

		for i in range(space):
			for j in range(space):
				self.test.append([i,j])

		self.buildBiasEngine()


	def inferSummary(self, state, action):
		self.inferLikelihood(state, action)
		self.inferPosterior(state, action)
		self.expectedPosterior()
		# self.plotDistributions()

	def buildBiasEngine(self):
		""" 
			Simulates MDPs with varying bias to build a bias inference engine.
		"""

		print "Loading MDPs...\n"

		# Unnecessary progress bar for terminal
		bar = pyprind.ProgBar(len(self.test))
		for i in self.test:
			self.sims.append(GridWorld(i[0], i[1]))
			bar.update()

		print "\nDone loading MDPs..."


	def inferLikelihood(self, state, action):
		"""
			Uses inference engine to inferBias predicated on an agents'
			actions and current state.
		"""

		self.state = state
		self.action = action

		self.likelihood = list()
		for i in range(len(self.sims)):
			self.likelihood.append(self.sims[i].policy[state][action])


	def inferPosterior(self, state, action):
		"""
			Uses inference engine to compute posterior probability from the 
			likelihood and prior (beta distribution).
		"""

		# Beta Distribution
		# self.prior = np.linspace(.01,1.0,101)
		# self.prior = beta.pdf(self.prior,1.4,1.4)
		# self.prior /= self.prior.sum()

		# Shifted Exponential
		# self.prior = np.zeros(101)
		# for i in range(50):
		# 	self.prior[i + 50] = i * .02
		# self.prior[100] = 1.0
		# self.prior = expon.pdf(self.prior)
		# self.prior[0:51] = 0
		# self.prior *= self.prior
		# self.prior /= self.prior.sum()

		# # Shifted Beta
		# self.prior = np.linspace(.01,1.0,101)
		# self.prior = beta.pdf(self.prior,1.2,1.2)
		# self.prior /= self.prior.sum()
		# self.prior[0:51] = 0

		# Uniform
		self.prior = np.zeros(len(self.sims))
		self.prior = uniform.pdf(self.prior)
		self.prior /= self.prior.sum()
		# self.prior[0:51] = 0


		self.posterior = self.likelihood * self.prior
		self.posterior /= self.posterior.sum()


	def plotDistributions(self):

		# Plotting Posterior
		# plt.figure(1)
		# plt.subplot(221)
		plt.plot(np.linspace(0,.99,100), self.posterior, '.')
		plt.xticks(np.linspace(.01,.99,100), np.arange(0,101,1))
		plt.ylabel('P(Action={}|State={})'.format(self.action, self.state))
		plt.xlabel('Bias')
		plt.title('Posterior Probability for Bias')

		# Plotting Likelihood
		# plt.subplot(222)
		# plt.plot(np.linspace(.01,.99,100),self.likelihood)
		# plt.ylabel('P(Action={}|State={})'.format(self.action,self.state))
		# plt.xlabel('Bias')
		# plt.title('Likelihood for Actions, States')

		# # Plotting Prior
		# plt.subplot(223)
		# plt.plot(np.linspace(.01,.99,100), self.prior)
		# plt.ylabel('P(Bias)')
		# plt.xlabel('Bias')
		# plt.title('Prior Probability')
		# plt.tight_layout()
		plt.show()


	def expectedPosterior(self):
		"""
			Calculates expected value for the posterior distribution.
		"""
		expectation_a = 0
		expectation_b = 0
		x = range(len(self.posterior))

		for i in range(len(self.posterior)):
			expectation_a += self.test[i][0] * infer.posterior[i]
			expectation_b += self.test[i][1] * infer.posterior[i]

		print "Expectation of Goal A: {}".format(expectation_a)
		print "Expectation of Goal B: {}".format(expectation_b)










