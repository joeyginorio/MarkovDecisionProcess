# Joey Velez-Ginorio
# Hypothesis Implementation
# ---------------------------------

import numpy as np
import copy
from collections import OrderedDict
from GridWorld import GridWorld
from Grid import Grid
import itertools


class Hypothesis():
	"""

		Provides functions to generate hypotheses from primitives and objects
		in some GridWorld.

		Primitives:
			- And
			- Or
			- Then

	"""
	def __init__(self, grid, hypothesis=None):		
		self.grid = grid


	def HGenerator(self):
		pass


	def eval(self, graphList):
		"""
			Takes in a list of graphStrings, then evaluates by:

				1.) Linking graphs to a start node
				2.) Finding minimum cost graphString
				3.) Removing repeated nodes e.g. 'AAB' -> 'AB'
				4.) Returning the optimal path from a series of graphs
		"""

		# Attach graphs to start node
		graphList = self.linkGraphStrings(graphList)

		# Find cheapest path
		graphString = self.minCostGraphString(graphList)

		# Remove redundant nodes, e.g. 'AAB' -> 'AB'
		graphString = ''.join(ch for ch, _ in itertools.groupby(graphString))

		return graphString 


	def buildDistanceMatrix(self):
		"""
			Using GridWorld, creates a distance matrix, detailing the costs 
			to go from any object to another.
		"""

		# Initialize distance matrix
		dist = np.zeros([len(self.grid.objects), len(self.grid.objects)])

		# For each object in gridworld
		for i in range(len(self.grid.objects)):
			for j in range(len(self.grid.objects)):

				# Compute distance from each object to another
				dist[i][j] = self.objectDist(self.grid.objects.keys()[j],
					self.grid.objects.keys()[i])

		# Hold onto distance as instance variable
		self.dist = dist

	def objectDist(self, start, obj):
		"""
			Return cost of going to some object
		"""

		# Generate a grid that only cares about 
		# getting to the input 'obj'
		objectGrid = copy.deepcopy(self.grid)
		objValue = objectGrid.objects[obj] 
		objectGrid.objects.clear()
		objectGrid.objects[obj] = objValue

		# Simulate GridWorld where only goal is obj
		objectWorld = GridWorld(objectGrid, [10])
		startCoord = self.grid.objects[start]

		# Count num. of steps to get to obj from start, that is the distance
		dist = objectWorld.simulate(objectWorld.coordToScalar(startCoord))

		return dist 

	def Or(self, A, B):
		"""
			Primitive function to do the 'Or' operation. 
			Essentially throws the contents of A and B into
			one list (of subgraphs). 

			e.g. A:['A'], B:['A,B'] ; Or(A,B):['A','A','B']
		"""

		# If input is a char, turn into numpy array
		if type(A) is not np.ndarray:
			A = np.array([A])
		if type(B) is not np.ndarray:
			B = np.array([B])

		return np.append(A,B)
		 

	def Then(self, A, B):
		"""
			Primitive function to do the 'Then' operation.
			Adds every possible combination of A->B for all content
			within A and B to a list. 

			e.g. A:['A'], B:['A,B'] ; Then(A,B):['AA','AB']
		"""

		# If input is a char, turn into numpy array
		if type(A) is not np.ndarray:
			A = np.array([A])
		if type(B) is not np.ndarray:
			B = np.array([B])

		# C will hold all combinations of A->B
		C = np.array([])
		for i in range(len(A)):
			for j in range(len(B)):
				C = np.append(C, A[i] + B[j])

		return C

	def And(self, A, B):
		"""
			Primitive function to do the 'And' operation.
			Defined as a composition of Or and Then.

			And(A,B) = Or(Then(A,B),Then(B,A))
		"""

		# If input is a char, turn into numpy array
		if type(A) is not np.ndarray:
			A = np.array([A])
		if type(B) is not np.ndarray:
			B = np.array([B])

		return self.Or(self.Then(A,B), self.Then(B,A))


	def minCostGraphString(self, graphList):
		"""
			Considers multiple graphStrings, returns the one
			that costs the least. 

			e.g. graphList = ['AB','AC'], 'AB'=2 'AC'=4
			returns 'AB'
		"""

		# Cost for each graphString in graphList
		costGraphList = np.zeros(len(graphList))

		# Cycle through the list, calculate cost
		for i, graphString in enumerate(graphList):
			costGraphList[i] = self.costGraphString(graphString)

		# Return cheapest graphString
		return graphList[np.argmin(costGraphList)]


	def linkGraphStrings(self, graphList):
		"""
			Join all the graphStrings into one tree, by 
			attaching 'S', the start node, to all graphs.
		"""
		return ['S' + graphString for graphString in graphList]

	def costGraphString(self, graphString):
		"""
			Iterates through a graphString and computes the
			cost.
		"""

		# Check distance between 2 Goals at a time, add to
		# running sum of cost. e.g. 'ABC' = cost('AB') + cost('BC')
		cost = 0
		for i in range(len(graphString)):
			
			# If substring has only one char, stop computing cost
			if len(graphString[i:i+2]) == 1:
				break

			cost += self.costEdge(graphString[i:i+2])

		return cost


	def costEdge(self, edgeString):
		"""
			Computes cost of an edge in the graphString.
			An edge is any two adjacent characters in 
			a graphString e.g. 'AB' in 'ABCD'
 
		"""

		# Find index of object, to use for indexing distance matrix
		objIndex1 = self.grid.objects.keys().index(edgeString[0])
		objIndex2 = self.grid.objects.keys().index(edgeString[1])

		return self.dist[objIndex1][objIndex2]

