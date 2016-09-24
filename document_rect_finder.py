u"""Механизм поиска прямоугольника с документом. За основу взят код отюсда:
https://github.com/opencv/opencv/blob/master/samples/python/squares.py"""

import numpy as np
import cv2
from debug_log import LOG
from utils import  order_points

def write_debug_image_with_countours( image, message, countours ):
	u"""Выводит в лог изображение с контурами"""
	if LOG.is_enabled():
		if len( image.shape ) == 2:
			image_copy = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
		else:
			image_copy = image.copy()
		cv2.drawContours( image_copy, countours, -1, (0, 255, 0), 3 )
		LOG.write( message + " COUNTOURS(" + str( len( countours ) ) + ")" , image_copy )

def angle_cos(p0, p1, p2):
	u"""Косинус между двумя векторами p1p0 и p1p2"""
	d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
	return abs( np.dot(d1, d2) / np.sqrt( np.dot(d1, d1)*np.dot(d2, d2) ) )

def get_image_rect(image):
	u"""Возвращает контур всего изображения"""
	return np.array([[0,0],[image.shape[1],0],[image.shape[1],image.shape[0]],[0,image.shape[0]]])

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

	pts = order_points( rect )

	s1 = get_tri_square( [ pts[0], pts[1], pts[2] ] )
	s2 = get_tri_square( [ pts[2], pts[3], pts[0] ] )

	return s1 + s2

def find_suitable_rects(bin_image):
	u"""Ищет четырех угольники на бинаризованном изображении и филтрует их"""
	rects = []
	supricious_rects = []
	
	# Считаем отсечения слишком больших и маленьких четырехугольников
	image_square = get_rect_square(get_image_rect(bin_image))
	good_rect_square_treshold = 0.85 * image_square
	supricious_rect_square_treshold = 0.95 * image_square
	min_rect_square = 0.15 * image_square

	# Ищем контуры
	_, contours, hierarchy = cv2.findContours(bin_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	# Оцениваем контуры
	for cnt in contours:
		cnt_len = cv2.arcLength(cnt, True)
		cnt = cv2.approxPolyDP(cnt, 0.02*cnt_len, True)
		if len(cnt) == 4 and cv2.contourArea(cnt) > 1000 and cv2.isContourConvex(cnt):
			cnt = cnt.reshape(-1, 2)
			max_cos = np.max([angle_cos( cnt[i], cnt[(i+1) % 4], cnt[(i+2) % 4] ) for i in range(4)])
			rect_square = get_rect_square( cnt )

			if rect_square < min_rect_square:
				continue

			if max_cos < 0.3 and good_rect_square_treshold > rect_square:
				rects.append( cnt )
			elif max_cos < 0.4 and supricious_rect_square_treshold > rect_square:
				supricious_rects.append( cnt )

	return rects, supricious_rects

def find_rects_enum_binarization( gray_image, binarization_step ):
	u"""Ищет прямоугольники, перебирая различные бинаризации по порогу с шагом binarization_step"""
	rects = []
	supricious_rects = []

	for thrs in range(0, 255, binarization_step):
			if thrs == 0:
				bin = cv2.Canny(gray_image, 0, 50, apertureSize=5)
				bin = cv2.dilate(bin, None)
			else:
				retval, bin = cv2.threshold(gray_image, thrs, 255, cv2.THRESH_BINARY)
			rects_to_extend,supricious_rects_to_extend = find_suitable_rects( bin )
			write_debug_image_with_countours( bin, "[RECTS FINDER] BINARIZATION WITH TRESHOLD " +
				str( thrs ), rects_to_extend )
			rects.extend( rects_to_extend )
			if len(supricious_rects_to_extend) > 0:
				write_debug_image_with_countours( bin, "[RECTS FINDER] SUPRICIOUS BINARIZATION WITH TRESHOLD " +
					str( thrs ), supricious_rects_to_extend )
			supricious_rects.extend( supricious_rects_to_extend )

	return rects, supricious_rects

def find_rects( image ):
	u"""Ищем четырехугольники на сером изображении"""

	# Строим серое изображние
	gray_image = cv2.cvtColor( image, cv2.COLOR_BGR2GRAY )
	LOG.write( "[RECTS FINDER] grayed", gray_image )

	# размытие улучшает поиск квадратов
	blured_img = cv2.GaussianBlur(gray_image,(5,5),0)
	LOG.write( "[RECTS FINDER] blured", blured_img )

	# Бинаризация с поиском порога методом Оцу
	_,bin = cv2.threshold(blured_img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	bin = cv2.dilate(bin, None)
	rects, supr_rects = find_suitable_rects( bin )

	write_debug_image_with_countours( bin,"[RECTS FINDER] OTSU", rects )
	if len(supr_rects) > 0:
		write_debug_image_with_countours( bin,"[RECTS FINDER] SUPRICIOUS OTSU", supr_rects )

	# Если на нашли на бинаризованном ОЦУ изображении прямоугольники, пробуем перебирать различные пороги
	# в ручную
	if len(rects) == 0:
		rects, sr = find_rects_enum_binarization( blured_img, 128 )
		supr_rects.extend( sr )

	if len(rects) == 0:
		rects, sr = find_rects_enum_binarization( blured_img, 26 )
		supr_rects.extend( sr )

	# Делать нечего. Прямоугольник - вся картинка
	if len( rects ) == 0:
		if len( supr_rects ) != 0:
			rects = supr_rects
		else:
			rects = [get_image_rect( image )]

	# Выводим найденные контуры
	write_debug_image_with_countours( image, "[RECTS FINDER] RESULT RECTS", rects )

	return rects


def find_document_rect( image ):
	u"""Ищем прямоугольник с документом на картинке"""

	#Алгоритм такой - получаем все прямоугольники и выбираем с максимальной площадью
	rects = find_rects( image )
	squares = [get_rect_square(r) for r in rects]
	max_index = squares.index( max( squares ) )

	write_debug_image_with_countours( image, "[RECTS FINDER] finded document rect", [rects[max_index]] )

	return rects[max_index]