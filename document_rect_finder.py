u"""Механизм поиска прямоугольника с документом. За основу взят код отюсда:
https://github.com/opencv/opencv/blob/master/samples/python/squares.py"""

import numpy as np
import cv2
from debug_log import LOG

def angle_cos(p0, p1, p2):
	u"""Косинус между двумя векторами p1p0 и p1p2"""
	d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
	return abs( np.dot(d1, d2) / np.sqrt( np.dot(d1, d1)*np.dot(d2, d2) ) )

def find_rects( image ):
	u"""Ищем четырехугольники на сером изображении"""

	# Строим серое изображние
	gray_image = cv2.cvtColor( image, cv2.COLOR_BGR2GRAY )
	LOG.write( "[SQUARE FINDER] grayed", gray_image )

	# размытие улучшает поиск квадратов
	blured_img = cv2.GaussianBlur(gray_image,(5,5),0)
	LOG.write( "[SQUARE FINDER] blured", blured_img )

	rects = []

	# Бинаризация с поиском порога методом Оцу
	_,bin = cv2.threshold(blured_img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	LOG.write( "[SQUARE FINDER] binarized", bin )
	
	# Ищем контуры
	bin, contours, hierarchy = cv2.findContours(bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	# Оцениваем контуры
	for cnt in contours:
		cnt_len = cv2.arcLength(cnt, True)
		cnt = cv2.approxPolyDP(cnt, 0.02*cnt_len, True)
		if len(cnt) == 4 and cv2.contourArea(cnt) > 1000 and cv2.isContourConvex(cnt):
			cnt = cnt.reshape(-1, 2)
			max_cos = np.max([angle_cos( cnt[i], cnt[(i+1) % 4], cnt[(i+2) % 4] ) for i in range(4)])
			if max_cos < 0.1:
				rects.append(cnt)

	# Выводим найденные контуры
	if LOG.is_enabled():
		image_copy = image.copy()
		cv2.drawContours( image_copy, rects, -1, (0, 255, 0), 3 )
		LOG.write( "[SQUARE FINDER] finded countours", image_copy )

	return rects

# Расчитываем площадь треугольника
def get_tri_square( tri ):
	assert len( tri ) == 3, "Invalid verteces count"

	# Длины сторон
	a = np.sqrt( np.dot(tri[1] - tri[0], tri[1]-tri[0] ) )
	b = np.sqrt( np.dot(tri[2] - tri[1], tri[2]-tri[1] ) )
	c = np.sqrt( np.dot(tri[0] - tri[2], tri[0]-tri[2] ) )

	# Полупериметр
	p = ( a + b + c ) / 2

	# Площадь:
	return np.sqrt( p * ( p - a ) * ( p - b ) * ( p - c ) )

# Расчитывает площадь четырех угольника
def get_rect_square( rect ):
	assert len( rect ) == 4, "Invalid verteces count"

	s1 = get_tri_square( [ rect[0], rect[1], rect[2] ] )
	s2 = get_tri_square( [ rect[2], rect[3], rect[0] ] )

	return s1 + s2

def find_document_rect( image ):
	u"""Ищем прямоугольник с документом на картинке"""

	#Алгоритм такой - получаем все прямоугольники и выбираем с максимальной площадью
	rects = find_rects( image )
	squares = [get_rect_square(r) for r in rects]
	max_index = squares.index( max( squares ) )

	if LOG.is_enabled():
		image_copy = image.copy()
		cv2.drawContours( image_copy, [rects[max_index]], -1, (0, 255, 0), 3 )
		LOG.write( "[SQUARE FINDER] finded document rect", image_copy )

	return rects[max_index]