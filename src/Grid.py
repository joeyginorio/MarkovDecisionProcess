# Joey Velez-Ginorio
# Gridworld Implementation
# ---------------------------------

import numpy as np
from collections import OrderedDict

class Grid():
	"""

		Defines the necessary environment elements for several variations
		of GridWorld to be easily constructed by GridWorld().

	"""

	def __init__(self, grid='bookGrid'):
		self.row = 0
		self.col = 0


		self.objects = OrderedDict()
		self.walls = list()
		

		if grid == 'bookGrid':
			self.getBookGrid()

		elif grid == 'testGrid':
			self.getTestGrid()


	def setGrid(self, fileName):
		""" 
			Initializes grid to the desired gridWorld configuration.
		"""
		gridBuffer = np.loadtxt(fileName, dtype=str)

		numObjects = int(gridBuffer[0])
		objectNames = list()
		objects = list()

		for i in range(numObjects):
			objectNames.append(gridBuffer[i+1])

		gridBuffer = gridBuffer[(numObjects+1):]

		self.row = len(gridBuffer)
		self.col = len(gridBuffer[0])

		gridMatrix = np.empty([self.row,self.col], dtype=str)

		for i in range(self.row):
			gridMatrix[i] = list(gridBuffer[i])

		objects = zip(*np.where(gridMatrix == 'O'))
		self.walls = zip(*np.where(gridMatrix == 'W'))
		start = zip(*np.where(gridMatrix == 'S'))[0]


		self.objects['start'] = start
		for i, o in enumerate(objects):
			self.objects[objectNames[i]] = o


	def getBookGrid(self):
		""" 
			Builds the canonical gridWorld example from the Sutton,
			Barto book.
		"""
		fileName = 'gridWorlds/bookGrid.txt'
		self.setGrid(fileName)
		
	def getTestGrid(self):
		"""
			Builds a test grid, use this to quickly try out different
			gridworld environments. Simply modify the existing testGrid.txt
			file.
		"""
		fileName = 'gridWorlds/testGrid.txt'
		self.setGrid(fileName)