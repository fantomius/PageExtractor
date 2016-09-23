u"""Вырезатель прямоугольника из изображения по 4-м точкам"""

import numpy as np
import cv2
from debug_log import LOG

def order_points(pts):
	u"""Переупорядочивает точки в прямоугольнике так, чтобы первой был левый верхний угол, второй - правый верхний
	третьей - правый нижний и четвертой - левый нижний угол"""
	assert len(pts) == 4, "Invalid points count"

	rect = np.zeros( (4, 2), dtype="float32" )
	s = pts.sum( axis = 1 )
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]

	diff = np.diff(pts, axis=1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]

	return rect

def extract_rect(image, pts):
	u"""Извлекает прямоугольник pts из изображения image, применяя к нему преобразование исправления перспективы"""
	rect = order_points( pts )

	(tl, tr, br, bl) = rect
	maxWidth = max( int( np.linalg.norm( br - bl ) ), int( np.linalg.norm( tr - tl ) ) )
	maxHeight = max( int( np.linalg.norm( br - tr ) ), int( np.linalg.norm( bl - tl ) ) )

	dst = np.array([[0,0],[maxWidth-1,0],[maxWidth-1, maxHeight-1],[0, maxHeight-1]], dtype="float32")

	transform = cv2.getPerspectiveTransform( rect, dst )
	warped = cv2.warpPerspective( image, transform, (maxWidth, maxHeight ) )

	LOG.write( "[RECTANGLE EXTRACTOR] EXTRACTED", warped )

	return warped