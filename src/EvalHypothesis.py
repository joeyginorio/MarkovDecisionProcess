# Joey Velez-Ginorio
# Gridworld Implementation
# ---------------------------------

import numpy as np
import copy
from collections import OrderedDict
from GridWorld import GridWorld
from Grid import Grid

class Hypothesis():
	"""

	"""
	def __init__(self, objects, primitives):
		"""
			Write a function to create a hypothesis using some grammar
		"""
		self.objects = objects
		self.primitives = primitives
		self.cost = None
		self.policy = None



class EvalHypothesis():
	"""

	"""
	def __init__(self, grid, hypothesis=None):
		self.grid = grid
		self.masterPolicy = None
		self.hypothesis = hypothesis
		self.objectCost = dict()
		self.dist = None

		self.buildDistanceMatrix()

	def buildDistanceMatrix(self):
		"""

		"""
		dist = np.zeros([len(self.grid.objects), len(self.grid.objects)])

		for i in range(len(self.grid.objects)):

			for j in range(len(self.grid.objects)):


				dist[i][j] = self.objectDist(self.grid.objects.keys()[j],
					self.grid.objects.keys()[i])

		self.dist = dist

	def objectDist(self, start, obj):
		"""
			Return cost of going to some object
		"""
		objectGrid = copy.deepcopy(self.grid)
		objValue = objectGrid.objects[obj] 
		objectGrid.objects.clear()
		objectGrid.objects[obj] = objValue

		objectWorld = GridWorld(objectGrid, [10])
		startCoord = self.grid.objects[start]

		dist = objectWorld.simulate(objectWorld.coordToScalar(startCoord))

		return dist 

	def Or(self, A, B):
		"""

		"""
		return np.append(A,B)
		 

	def Then(self, A, B):
		"""

		"""
		C = np.array([])

		for i in range(len(A)):
			for j in range(len(B)):
				C = np.append(C, A[i] + B[j])


		return C

	def And(self, A, B):
		"""

		"""
		return self.Or(self.Then(A,B), self.Then(B,A))

	def linkGraphStrings(self, graphList):
		"""

		"""
		return ['S' + graphString for graphString in graphList]

	def minCostGraphString(self, graphList):
		"""

		"""
		costGraphList = np.zeros(len(graphList))

		for i, graphString in enumerate(graphList):
			costGraphList[i] = self.costGraphString(graphString)

		return graphList[np.argmin(costGraphList)]


	def costGraphString(self, graphString):
		"""

		"""
		cost = 0

		for i in range(len(graphString)):
			
			if len(graphString[i:i+2]) == 1:
				break

			cost += self.costEdge(graphString[i:i+2])

		return cost


	def costEdge(self, edgeString):
		"""

		"""
		objIndex1 = self.grid.objects.keys().index(edgeString[0])
		objIndex2 = self.grid.objects.keys().index(edgeString[1])

		return self.dist[objIndex1][objIndex2]

