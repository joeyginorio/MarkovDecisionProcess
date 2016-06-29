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
		
		# Sutton, Barto canonical example
		if grid == 'bookGrid':
			self.getBookGrid()

		# GridWorld implementation for testing, modify testGrid.txt
		elif grid == 'testGrid':
			self.getTestGrid()


	def setGrid(self, fileName):
		""" 
			Initializes grid to the desired gridWorld configuration.
		"""
		
		# Load in the .txt gridworld
		gridBuffer = np.loadtxt(fileName, dtype=str)

		# Find out how many objects to look for
		numObjects = int(gridBuffer[0])

		objectNames = list()
		objects = list()

		# Store the names for each object
		for i in range(numObjects):
			objectNames.append(gridBuffer[i+1])

		# Set gridbuffer to only contain the map
		gridBuffer = gridBuffer[(numObjects+1):]

		# Find row/col of map
		self.row = len(gridBuffer)
		self.col = len(gridBuffer[0])

		# Store the gridworld in a matrix
		gridMatrix = np.empty([self.row,self.col], dtype=str)
		for i in range(self.row):
			gridMatrix[i] = list(gridBuffer[i])

		# Store all object locations
		objects = zip(*np.where(gridMatrix == 'O'))

		# Store all wall locations
		self.walls = zip(*np.where(gridMatrix == 'W'))

		# Store object:location in dictionary
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