# Joey Velez-Ginorio
# MDP Implementation
# ---------------------------------
# - Includes BettingGame example


import matplotlib.pyplot as plt
import numpy as np
import random
import pyprind
from scipy.stats import beta
from scipy.stats import expon
from scipy.stats import uniform
from abc import ABCMeta
from abc import abstractmethod

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
	
	def __init__(self, states=None, actions=None, rewards=None, transitions=None, 
				discount=.99, tau=.01, epsilon=.01):

		self.s = np.array(states)
		self.a = np.array(actions)
		self.r = np.array(rewards)
		self.t = np.array(transitions)
		
		self.discount = discount
		self.tau = tau
		self.epsilon = epsilon

		# Value iteration will update this
		self.values = None
		self.policy = None

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


	def valueIteration(self):
		"""
			Performs value iteration to populate the values of all states in
			the MDP. 

			Params:
				- epsilon: Determines limit of convergence
		"""

		# Initialize V_0 to zero
		self.values = np.zeros(len(self.s))
		self.policy = np.zeros([len(self.s), len(self.a)])

		policy_switch = 0

		# Loop until convergence
		while True:

			# oldPolicy = np.argmax(self.policy, axis=1)
			# self.extractPolicy()
			# newPolicy = np.argmax(self.policy, axis=1)


			# if not np.array_equal(oldPolicy, newPolicy):
			# 	policy_switch += 1

			# print "Policy switch count: {}".format(policy_switch)

			# To be used for convergence check
			oldValues = np.copy(self.values)

			for i in range(len(self.s)-1):

				self.values[i] = self.r[i] + np.max(self.discount * \
							np.dot(self.t[i][:][:], self.values))

			# print "Convergence Measure: {}".format(np.max(np.abs(self.values - oldValues)))
			# print "-------------------------------------"
			# Check Convergence
			if np.max(np.abs(self.values - oldValues)) <= self.epsilon:
				break



	def extractPolicy(self):
		"""
			Extract policy from values after value iteration runs.
		"""

		self.policy = np.zeros([len(self.s),len(self.a)])

		for i in range(len(self.s)-1):

			state_policy = np.zeros(len(self.a))

			state_policy = self.r[i] + self.discount* \
						np.dot(self.t[i][:][:], self.values)

			# Softmax the policy			
			state_policy -= np.max(state_policy)
			state_policy = np.exp(state_policy / float(self.tau))
			state_policy /= state_policy.sum()

			self.policy[i] = state_policy

	def extractDeterministicPolicy(self):
		"""
			Extract policy from values after value iteration runs.
		"""
		self.policy = np.zeros(len(self.s))

		for i in range(len(self.s)-1):

			# Take max over all possible actions in state
			max_a = 0


			for j in range(len(self.a)):

				# Account for all possible states a particular action can take you to
				sum_nextState = 0
				for k in range(len(self.s)-1):
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

	def __init__(self, pHeads=.5, discount=.99, epsilon=.1, tau=.001):

		MDP.__init__(self,discount=discount,tau=tau,epsilon=epsilon)
		self.pHeads = pHeads
		self.setBettingGame(pHeads)
		self.valueIteration()
		self.extractPolicy()

	def isTerminal(self, state):
		"""
			Checks if MDP is in terminal state.
		"""
		return True if state is 100 or state is 0 else False

	def setBettingGame(self, pHeads=.5):

		""" 
			Initializes the MDP to the starting conditions for 
			the betting game. 

			Params:
				startCash = Amount of starting money to spend
				pHeads = Probability that coin lands on head
					- .5 for fair coin, otherwise choose bias

		"""

		# This is how much we're starting with
		self.pHeads = pHeads

		# Initialize all possible states
		self.s = np.arange(102)

		# Initialize possible actions
		self.a = np.arange(101)

		# Initialize rewards
		self.r = np.zeros(101)
		self.r[0] = -5
		self.r[100] = 10

		# Initialize transition matrix
		temp = np.zeros([len(self.s),len(self.a),len(self.s)])

		# List comprehension using tHelper to determine probabilities for each index
		self.t = [self.tHelper(i[0], i[1], i[2], self.pHeads) for i,x in np.ndenumerate(temp)]
		self.t = np.reshape(self.t, np.shape(temp))
		
		for x in range(len(self.a)):

		# Remembr to add -1 to value it, and policy extract
			# Send the end game states to the death state!
			self.t[100][x] = np.zeros(len(self.s))
			self.t[100][x][101] = 1.0
			self.t[0][x] = np.zeros(len(self.s))
			self.t[0][x][101] = 1.0

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
			return 1.0

		# If you bet more money than you have, no chance of any outcome
		elif y > x and x is not z:
			return 0

		# If you bet more money than you have, returns same state with 1.0 prob.
		elif y > x and x is z:
			return 1.0

		# Chance you lose
		elif x - y is z:
			return 1.0 - pHeads

		# Chance you win
		elif x + y is z:
			return pHeads

		# Edge Case: Chance you win, and winnings go over 100
		elif x + y > z and z is 100:
			return pHeads


		else:
			return 0 

		return 0
 

class InferenceMachine():
	"""
		Conducts inference via MDPs for the BettingGame.
	"""
	def __init__(self):
		self.sims = list()

		self.likelihood = None
		self.posterior = None
		self.prior = None

		self.e = None

		self.buildBiasEngine()


	def inferSummary(self, state, action):
		self.inferLikelihood(state, action)
		self.inferPosterior(state, action)
		print "Expected Value of Posterior Distribution: {}".format(
			self.expectedPosterior())
		self.plotDistributions()

	def buildBiasEngine(self):
		""" 
			Simulates MDPs with varying bias to build a bias inference engine.
		"""

		print "Loading MDPs...\n"

		# Unnecessary progress bar for terminal
		bar = pyprind.ProgBar(len(np.arange(0,1.01,.01)))
		for i in np.arange(0,1.01,.01):
			self.sims.append(BettingGame(i))
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
		self.prior = np.linspace(.01,1.0,101)
		self.prior = uniform.pdf(self.prior)
		self.prior /= self.prior.sum()
		self.prior[0:51] = 0


		self.posterior = self.likelihood * self.prior
		self.posterior /= self.posterior.sum()


	def plotDistributions(self):

		# Plotting Posterior
		plt.figure(1)
		plt.subplot(221)
		plt.plot(np.linspace(.01,1.0,101), self.posterior)
		plt.ylabel('P(Action={}|State={})'.format(self.action, self.state))
		plt.xlabel('Bias')
		plt.title('Posterior Probability for Bias')

		# Plotting Likelihood
		plt.subplot(222)
		plt.plot(np.linspace(.01,1.0,101),self.likelihood)
		plt.ylabel('P(Action={}|State={})'.format(self.action,self.state))
		plt.xlabel('Bias')
		plt.title('Likelihood for Actions, States')

		# Plotting Prior
		plt.subplot(223)
		plt.plot(np.linspace(.01,1.0,101), self.prior)
		plt.ylabel('P(Bias)')
		plt.xlabel('Bias')
		plt.title('Prior Probability')
		plt.tight_layout()
		plt.show()


	def expectedPosterior(self):
		"""
			Calculates expected value for the posterior distribution.
		"""
		expectation = 0
		x = np.linspace(.01,1.0,101)

		for i in range(len(self.posterior)):
			expectation += self.posterior[i] * x[i]

		return expectation


# infer = InferenceMachine()


"""
(.8 discount expected values)

(20,10) -> .769 | .75, .75, .9, .75, .8, .75, .7, .8		| .7749 Close
(20,5) -> .668 | .65, .625, .6, .66, .65, .63, .65, .75 	| .6519	Close
(20,4) -> .607 | .60, .5725, .58, .63, .6, .6, .6, .6		| .5978 Close
(20,1) -> .591 | .5, .53125, .51, .5, .5, .5, .5, .5		| .5052 Eh

(40,5) -> .585  | .6, .5725, .65, .55, .6, .56, .5, .6		| .5791 Close
(40,10) -> .650 | .65, .625, .7, .6, .65, .63, .55, .65		| .6319 Close
(40,20) -> .777 | .75, .75, .95, .7, .8, .75, .75, .75		| .7749 Close
(40,40) -> .646 | 1.0, 1.0, 1.0, 1.0, .95, .9, 1.0, .9		| .9688 Eh

(80,1) -> .581  | .5, .515625, .51, .5, .65, .5, .5, .5		| .522 Eh
(80,5) -> .578  | .55, .53125, .55, .56, .75, .65, .5, .6	| .586 Close
(80,10) -> .605 | .6, .5725, .6, .67, .85, .75, .6, .7		| .668 Eh 
(80,20) -> .683 | .65, .625, .65, .75, .95, .9, .65, .8		| .749 Eh
"""

"""

Model can't capture intuition that betting all your money means you
probably are going to win. I can modify it to capture that intuition, but 
then the rest of the model breaks.

 x axis - model judgement
 y axis - participant judgement 

 """

