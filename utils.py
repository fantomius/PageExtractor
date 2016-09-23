u"""Вспомогательные утилиты"""

import numpy as np
import cv2

def resize_image_to_fit(image, max_width, max_height):
	u"""Перемасштабирует картинку, сохраняя aspect_ration, так, чтобы она влезала в max_width x max_height"""

	# OpenCv почему-то хранит высоту в shape[0], а ширину в shape[1]
	img_width, img_height = image.shape[1], image.shape[0]
	assert img_width > 0 and img_height > 0, "Image can't have 0 width or height"

	scale_coefts = ( max_width / img_width, max_height / img_height )
	target_scale = min( scale_coefts[0], scale_coefts[1] )
	dim = (int( target_scale * img_width ), int( target_scale * img_height ) )

	return cv2.resize(image, dim)

def scale_image( image, scale ):
	u"""Перемасштабирует картинку, умножая каждое измерение на scale"""
	dim = ( int( scale * image.shape[1] ), int( scale * image.shape[0] ) )
	return cv2.resize(image, dim)
