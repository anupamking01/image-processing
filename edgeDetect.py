'''@Pranjal Verma'''

import numpy as np
import cv2, math, copy
from numba import jit

IMG_PATHS = ['vangogh.jpg', 'Valve.png', 'Bikesgray.jpg', 'ironman.png', 'coins.jpg']

MAX_VAL = 200 # 200
MIN_VAL = 50 # 50

EDGE_THICKNESS_FACTOR = 2
IMG_SHOW_TIME = 7000

def SobelOperator(axis='x'):
	'''Sobel Operator matrices; horizontal and vertical'''

	if axis == 'x':
		return np.matrix([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
	elif axis == 'y':
		return np.matrix([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
	else:
		raise Exception('Bad direction!')

def getImage(path):
	'''Load image and preprocess'''

	img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
	img = cv2.GaussianBlur(img, (5, 5), 0)

	newRow, newCol = img.shape
	newRow, newCol = newRow - newRow%3, newCol - newCol%3

	return img, newRow, newCol

def GradientDirection(Gx, Gy):
	'''Gives local coordinates in gradient direction'''

	# Possible directions
	anglesCoords = {0:[(0, 1), (0, -1)],
					45:[(-1, 1), (1, -1)],
					90:[(-1, 0), (1, 0)],
					135:[(-1, -1), (1, 1)]}

	if Gx == 0:
		w = math.inf
	else:
		w = Gy/Gx

	gradAngle = math.degrees(math.atan(w))
	if gradAngle < 0:
		gradAngle += 360

	# Getting closest angle
	gradDir = min(anglesCoords, key=lambda x:abs(x - gradAngle))
	return anglesCoords[gradDir]

def postProcess(G, Gx, Gy):
	'''Refines strong edges, removes weak edges'''
	row, col = G.shape

	# Non-Edge Suppression
	G_PostProcess = copy.deepcopy(G)
	for i in range(1, row - 1, EDGE_THICKNESS_FACTOR):
		for j in range(1, col - 1, EDGE_THICKNESS_FACTOR):
			localCoords = GradientDirection(Gx[i, j], Gy[i, j])

			x1, y1 = i + localCoords[0][0], j + localCoords[0][1]
			x2, y2 = i + localCoords[1][0], j + localCoords[1][1]
			if G[i, j] != max(G[i, j], max(G[x1, y1], G[x2, y2])):
				G_PostProcess[i, j] = 0

	# Strong edge Highlighting and BG darkening
	for i in range(0, row):
		for j in range(0, col):
			if G_PostProcess[i, j] <= MIN_VAL:
				G_PostProcess[i, j] = 0

			elif G_PostProcess[i, j] >= MAX_VAL:
				G_PostProcess[i, j] = 225

	return G_PostProcess

def getGradientMatrix(Gx, Gy):
	'''Computes final gradient matrix for given image'''
	row, col = Gx.shape

	G = np.zeros((row, col), dtype=np.uint8)
	for i in range(0, row):
		for j in range(0, col):
			G[i, j] = np.uint8(math.sqrt(pow(Gx[i, j], 2) + pow(Gy[i, j], 2)))

	return G

@jit(nopython=True)
def convolve(G, S):
	'''Convolve two 2d matrices, keeping same shape as G'''
	row, col = G.shape

	# Ignoring boundries, center-aligned convMatrix
	convMatrix = np.zeros((row, col), dtype=np.int64)
	for i in range(1, row - 1):
		for j in range(1, col - 1):
			# Element-wise multiplication and summing
			convMatrix[i, j] = np.sum(np.multiply(G[i-1:i+2, j-1:j+2], S))

	return convMatrix

def detectEdges(imgPath):
	'''Highlights edges in given image'''
	img, row, col = getImage(imgPath)

	# Detecting horizontal and vertical edges
	Gx = convolve(img, SobelOperator())
	Gy = convolve(img, SobelOperator('y'))

	# combining Gx and Gy to detect all edges
	G = getGradientMatrix(Gx, Gy)
	G_PostProcess = postProcess(G, Gx, Gy)

	# G_PostProcess = cv2.resize(G_PostProcess, (600, 600))
	cv2.imshow('Edge Detection', G_PostProcess)
	cv2.waitKey(IMG_SHOW_TIME)

	return

if __name__ == '__main__':
	for path in IMG_PATHS:
		detectEdges(path)
	cv2.destroyAllWindows()
