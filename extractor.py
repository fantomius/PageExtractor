u"""Класс эксрактора и метод его создания"""

import numpy as np
import cv2

from document_rect_finder import find_document_rect

class Extractor:
	def __init__(self):
		pass

	def process_image( self, _image ):
		u"""Возвращает картинку документа, извлеченную из фото"""
		image = _image.copy()

		document_rect = find_document_rect( image )
		cv2.drawContours( image, [document_rect], -1, (0, 255, 0), 3 )

		return image

def create_extractor():
	return Extractor()