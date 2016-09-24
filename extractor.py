u"""Класс эксрактора и метод его создания"""

import numpy as np
import cv2

from document_rect_finder import find_document_rect
from rect_extractor import extract_rect
import image_preprocessor as preproc

class Extractor:
	def __init__(self):
		pass

	def process_image( self, _image, use_preprocessing = True ):
		u"""Возвращает картинку документа, извлеченную из фото"""
		if use_preprocessing:
			image = preproc.preprocess_image( _image )
		else:
			image = _image.copy()

		document_rect = find_document_rect( image )
		document_image = extract_rect( image, document_rect )

		return document_image

def create_extractor():
	return Extractor()