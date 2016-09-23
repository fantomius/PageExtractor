u"""Вырезатель прямоугольника из изображения по 4-м точкам"""

import numpy as np
import cv2
from debug_log import LOG
from utils import order_points

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