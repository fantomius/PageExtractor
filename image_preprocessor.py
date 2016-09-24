u"""Функции для преподготовки изображения к обработке"""

import numpy as np
import cv2

from debug_log import LOG

def auto_contrast(image):
	u"""Повышение контраста и яркости изображения"""
	result = []
	for img in cv2.split( image ):
		# Повышаем контрастность
		clache = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
		result.append( clache.apply( img ) )

	return cv2.merge( result )

def sharpen(image):
	u"""Исправление небольших размытий"""
	result = []
	for img in cv2.split( image ):
		blured = cv2.Laplacian( img, cv2.CV_8U )
		result.append( cv2.addWeighted( img, 1.2, blured, -0.5, 0 ) )

	return cv2.merge( result )

def preprocess_image( image ):
	u"""Предобработка изображения"""
	result = image

	result = auto_contrast( result )
	LOG.write( "[PREPROCESSOR] CONTRAST", result)

	result = sharpen( result )
	LOG.write( "[PREPROCESSOR] SHARPEN", result )

	result = cv2.fastNlMeansDenoisingColored( image, None, 10, 10, 7, 21 )
	LOG.write( "[PREPROCESSOR] DENOISING", result )

	return result
