# PageExtractor
Приложение для выделения страницы текста с фотографии
<br/>
# Установка
Программа написана на Python 3 и зависит от следующих компонент:<br/>
1. numpy и PyQt4 (рекомендуется ставить дистрибутив Anaconda отсюда - https://www.continuum.io/downloads).<br/>
2. Нужен OpenCV для Python3. Скачать .whl отсюда: http://www.lfd.uci.edu/~gohlke/pythonlibs/#opencv,
установить при помощи pip install {путь_к_whl_файлу}<br/>
<br/>
# Запуск
Запускать main.py
<br/>
# Описание решения
В начале был написан алгоритм выделения прямоугольника с документом на изображении. За основу был взят алгоритм из примера OpenCV (https://github.com/opencv/opencv/blob/master/samples/python/squares.py). Он работает по следующему принципу:<br/>
1. Берется серая картинка изображения, немного размывается и после этого бинаризуется <br/>
2. На бинаризованной картинке средствами OpenCV происходит поиск контуров<br/>
3. Контуры фильтруются<br/>
	а) Сначала отсекаются контуры, которые не могут быть аппроксимированы прямоугольниками<br/>
	б) Затем отсекаются прямоугольные контуры, у которых слишом маленькие или большие углы между сторонами<br/>
<br/>
К этому алгоритму был добавлен ряд улучшений:<br/>
1. Бинаризация происходит различными способами:<br/>
	а) Сначала происходит бинаризация с выбором порога по ОЦУ https://ru.wikipedia.org/wiki/%D0%9C%D0%B5%D1%82%D0%BE%D0%B4_%D0%9E%D1%86%D1%83<br/>
	б) Если после бинаризации по ОЦУ мы не нашли контуры, то перебираются обычные бинаризации с разными порогами<br/>
2. Отсекаем контуры, которые имеют слишком маленькую площадь (меньше, чем 0.15 * площадь изображения)<br/>
3. Выделяем "хорошие" контуры и "подозрительные" контуры<br/>
	а) Хорошие контуры имеют косинус угла между сторонами не больше 0.3 и площадь не больше 0.85 от площади изображения<br/>
	б) Подозрительные контуры не удовлетворяют условиям хороших, но имеют косинус не больше 0.4 и площадь не больше .95<br/>
4. Если мы не нашли ни одного хорошего контура, то просматриваем подозрительные контуры<br/>
5. Если мы не нашли ни одного контура вообще, то тогда контур - всё изображение<br/>
<br/>
Из полученного множества контуров мы выделяем наиболее правдоподобный. Сейчас это реализовано предельно просто - мы выбираем максимальный по площади контур.<br/>
<br/>
Вся эта логика реализована в document_rect_finder.py.<br/>
<br/>
Далее было написано выделение картинки из этого прямоугольника. Алгоритм предельно простой - мы просто применяем преобразование восстановления перспективы и получаем картинку. Код расположен в файле rect_extractor.py<br/>
<br/>
После этого началась работа над различными преобразованиями картинки. В настоящее время выполняются следующие действия:<br/>
1. Адаптивно корректируется гистограмма, чтобы цвета располагались более-менее равномерно. Подробное описание - тут:
http://docs.opencv.org/3.1.0/d5/daf/tutorial_py_histogram_equalization.html (подраздел CLAHE)<br/>
2. Происходит коррекция резкости (небольшая), засчет вычитания картинки и её размытой копии<br/>
3. Происходит denoising картинки средствами OpenCV<br/>
<br/>
Код предобработки расположен в файле image_preprocessor.py<br/>
<br/>
Кроме того, были написаны вспомогательные классы:<br/>
1. Класс для вывода отладочных сообщений (debug_log.py)<br/>
2. Интерфейс пользователя - (gui.py)<br/>
<br/>
# Дальнейшие улучшения
Дальнейшие улучшения в основном будут связаны с предобработкой картинки<br/>
1. Текущая предобработка иногда помогает, иногда нет, но занимает очень много времени. Нужна выборка примеров, где она не помогает<br/>
2. Работа с motion_blur и расфокусировкой изображения. Убирать её можно при помощи Wiener Deconvolution (https://github.com/opencv/opencv/blob/master/samples/python/deconvolution.py), но нужно научиться определять параметры для него<br/>
3. Работа с засветами от вспышки и т.п.<br/>
4. Добавить качестенную оценку контуров<br/>


