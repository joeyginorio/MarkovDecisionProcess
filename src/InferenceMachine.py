# Joey Velez-Ginorio
# Gridworld Implementation
# ---------------------------------

from GridWorld import GridWorld
from Hypothesis import Hypothesis
from Grid import Grid
from scipy.stats import uniform
from scipy.stats import beta
from scipy.stats import expon
import numpy as np
import copy
import pyprind
import matplotlib.pyplot as plt


class InferenceMachine():
	"""

		Conducts inference over a hypothesis space defined via
		objects and primitives. 

		Can be used to generate all aspects of Bayes' Rule:
			- Prior
			- Likelihood
			- Posterior
		
	"""

	def __init__(self, grid, discount=.99, tau=.01, epsilon=.01):
		self.sims = list()

		# Key elements of Bayes' Rule
		self.likelihood = None
		self.posterior = None
		self.prior = None

		# Modify the GridWorld solver
		self.discount = discount
		self.tau = tau
		self.epsilon = epsilon

		self.grid = grid

		# Generate separate grids, one for each goal in the original map
		# This will be used later to generate subpolicies for each goal
		# e.g. A policy just for going to A, one just for B, etc.
		objectGrids = np.empty(len(self.grid.objects), dtype=object)
		for i in range(len(self.grid.objects)):
			objectGrids[i] = copy.deepcopy(self.grid)
			objValue = objectGrids[i].objects[self.grid.objects.keys()[i]] 
			objectGrids[i].objects.clear()
			objectGrids[i].objects[self.grid.objects.keys()[i]] = objValue

		self.objectGrids = objectGrids

		# Simulates the MDP's needed to begin the inference process
		# In this case, one for each object in the map
		# Subpolicies for each objectGrid done here.
		self.buildBiasEngine()


	def getStateActionVectors(self,start,actions):
		"""
			Generates the state vectors resulting from starting in some 
			state and taking a series of actions. Useful for testing 
			inference over state,action sequences (only have to come
			up with start state and actions, instead of having to manually
			calculate both state,action pairs each time).
		"""

		states = list()
		states.append(start)

		# Cycle through all actions, store the states they take you to
		for i in np.arange(0,len(actions)):

			# State that an action in current state takes me to
			nextState = self.sims[0].takeAction(self.sims[0].scalarToCoord(states[i]),actions[i])
			states.append(self.sims[0].coordToScalar(nextState))

		self.states = states
		self.actions = actions


	def getPolicySwitch(self, hypothesis, states):
		"""
			Generates a vector detailing, according to a hypothesis, when
			to switch policies when iterating across a vector of states.
		"""

		# State location of all goals/object in map
		goalStates = [self.sims[0].coordToScalar(goalCoord) for goalCoord in
					self.grid.objects.values()]

		# Create a dict, mapping goals->state_index
		goalIndex = dict()
		for i in range(len(goalStates)):
			goalIndex[self.grid.objects.keys()[i]] = goalStates[i]

		# Initialize policySwitch vector
		switch = np.empty(len(states), dtype=str)
		switchCount = 0

		# Iterate across states, if you hit current goalState, switch to next goalState
		# as your objective.
		# Essentially, if goals are 'ABC', stay in A until you hit A, then make B the goal
		for i, state in enumerate(states):
			if state == goalIndex[hypothesis[switchCount]] and switchCount + 1 < len(hypothesis):
				switchCount += 1

			switch[i] = hypothesis[switchCount]

		return switch


	def inferSummary(self, hypotheses, start, actions):
		"""
			Provide the prior, likelihood, and posterior distributions 
			for a set of hypotheses. 

			Utilizes Bayes' Rule, P(H|D) ~ P(D|H)P(H)

		"""

		# Add starting object to map
		self.grid.objects['S'] = tuple(self.sims[0].scalarToCoord(start))

		# Initialize the hypotheis generator
		self.H = Hypothesis(self.grid)

		# Setup the distance matrix, for calculating cost of hypotheses
		self.H.buildDistanceMatrix()

		# For each hypotheses, evaluate it and return the minCost graphString
		for i in range(len(hypotheses)):
			hypotheses[i] = self.H.eval(hypotheses[i])

		# Remove the 'S' node from each graph, no longer needed
		# since cost of each graph has been computed
		hypotheses = [hyp[1:] for hyp in hypotheses]
		self.hypotheses = hypotheses

		# Get state,action vectors to conduct inference over
		self.getStateActionVectors(start,actions)

		# Get policySwitch vector to know when to follow which policy
		self.policySwitch = list()
		for i in range(len(hypotheses)):
			self.policySwitch.append(self.getPolicySwitch(hypotheses[i], self.states))

		# Compute the likelihood for all hypotheses
		self.inferLikelihood(self.states, self.actions, self.policySwitch)

		# self.inferPosterior(state, action)
		# self.expectedPosterior()
		# self.plotDistributions()

	def buildBiasEngine(self):
		""" 
			Simulates the GridWorlds necessary to conduct inference.
		"""

		# Builds/solves gridworld for each objectGrid, generating policies for
		# each object in grid. One for going only to A, another just for B, etc.
		for i in range(len(self.objectGrids)):
			self.sims.append(GridWorld(self.objectGrids[i], [10], self.discount, self.tau, self.epsilon))
			


	def inferLikelihood(self, states, actions, policySwitch):
		"""
			Uses inference engine to inferBias predicated on an agents'
			actions and current state.
		"""

		self.states = states
		self.actions = actions
		self.likelihood = list()

		
		for i in range(len(policySwitch)):
			
			p = 1
			for j in range(len(policySwitch[0])-1):

				if states[j] == self.sims[0].coordToScalar(self.grid.objects[self.policySwitch[i][j]]):
					p *= self.sims[self.grid.objects.keys().index(policySwitch[i][j])].policy[self.sims[0].s[len(self.sims[0].s)-1]][actions[j]]
				
				else:
					p *= self.sims[self.grid.objects.keys().index(policySwitch[i][j])].policy[states[j]][actions[j]]

			self.likelihood.append(p)


	def inferPosterior(self, state, action, prior='uniform'):
		"""
			Uses inference engine to compute posterior probability from the 
			likelihood and prior (beta distribution).
		"""

		if prior == 'beta':
			# Beta Distribution
			self.prior = np.linspace(.01,1.0,101)
			self.prior = beta.pdf(self.prior,1.4,1.4)
			self.prior /= self.prior.sum()

		elif prior == 'shiftExponential':
			# Shifted Exponential
			self.prior = np.zeros(101)
			for i in range(50):
				self.prior[i + 50] = i * .02
			self.prior[100] = 1.0
			self.prior = expon.pdf(self.prior)
			self.prior[0:51] = 0
			self.prior *= self.prior
			self.prior /= self.prior.sum()

		elif prior == 'shiftBeta':
			# Shifted Beta
			self.prior = np.linspace(.01,1.0,101)
			self.prior = beta.pdf(self.prior,1.2,1.2)
			self.prior /= self.prior.sum()
			self.prior[0:51] = 0

		elif prior == 'uniform':
			# Uniform
			self.prior = np.zeros(len(self.sims))	
			self.prior = uniform.pdf(self.prior)
			self.prior /= self.prior.sum()


		self.posterior = self.likelihood * self.prior
		self.posterior /= self.posterior.sum()

	def expectedPosterior(self):
		"""
			Calculates expected value for the posterior distribution.
		"""
		expectation_a = 0
		expectation_b = 0
		aGreaterB = 0
		aLessB = 0
		aEqualB = 0

		x = range(len(self.posterior))

		for i in range(len(self.posterior)):

			e_a = self.test[i][0] * infer.posterior[i]
			e_b = self.test[i][1] * infer.posterior[i]

			expectation_a += e_a
			expectation_b += e_b

			# print "R_A: {}, R_B: {}".format(self.test[i][0], self.test[i][1])
			# print "E_a: {}".format(e_a)
			# print "E_b: {}\n".format(e_b)

			
			if self.test[i][0] > self.test[i][1]:
				aGreaterB += self.posterior[i]

			elif self.test[i][0] < self.test[i][1]:
				aLessB += self.posterior[i]

			elif self.test[i][0] == self.test[i][1]:
				aEqualB += self.posterior[i]
		
		# print aGreaterB


		print "Chance that agent prefers A over B: {}".format(aGreaterB)
		print "Chance that agent prefers B over A: {}".format(aLessB)
		print "Chance that agent prefers A and B equally: {}".format(aEqualB)

		print "Expectation of Goal A: {}".format(expectation_a)
		print "Expectation of Goal B: {}".format(expectation_b)




#######################################################################
#################### Testing ##########################################


testGrid = Grid('testGrid')
H = Hypothesis(testGrid)
infer = InferenceMachine(testGrid)

# Define starting state, proceeding actions
start = 8
actions = [0,0,3]

# Form hypotheses
hyps = list()
h1 = H.Then('A', 'B')
h2 = ['A']
h3 = H.Or('A','B')
h4 = ['B']
hyps = [h1,h2,h3,h4]

# Test Hypotheses
infer.inferSummary(hyps,start,actions)

print "\nHypotheses: {}".format(infer.hypotheses)
print "Likelihoods: {}".format(infer.likelihood)
print "States: {}".format(infer.states)
print "Actions: {}".format(infer.actions)


# Write a function for multinomial for each derivation, stop symbol
# write a recursive function till it calls a terminal function