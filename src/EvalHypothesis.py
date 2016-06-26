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

	def object(self, S, A):
		"""

		""" 

		# Calculate cost from S->A
		S_index = self.grid.objects.keys().index(S)
		A_index = self.grid.objects.keys().index(A)
		cost = self.dist[S_index][A_index]

		# Build subPolicy matrix from S->A
		subPolicy = np.zeros([len(evalH.grid.objects), len(evalH.grid.objects)-1])

		if A_index-1 < 0:
			pass

		else:
			subPolicy[S_index][A_index-1] = 1

		return np.array([cost, subPolicy, A], dtype=object)



	def Or(self, S, A, B):
		"""

		"""
		if type(S) is str:
			S = self.object(S,S)
		if type(A) is str:
			A = self.object(S[2][-1],A)
		if type(B) is str:
			B = self.object(S[2][-1],B)

		return A if A[0] < B[0] else B

	def Then(self, S, A, B):
		"""

		"""
		if type(S) is str:
			S = self.object(S,S)
		if type(A) is str:
			A = self.object(S[2][-1],A)
		if type(B) is str:
			B = self.object(A[2][-1],B)

		return A + B

 