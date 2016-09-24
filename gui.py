u"""Основные классы GUI"""

from PyQt4 import QtGui, QtCore
import numpy as np
import cv2

from utils import scale_image
from debug_log import LOG
from extractor import create_extractor

class ImageView(QtGui.QWidget):
	u"""Умеет показывать OpenCV изображение с масштабом"""
	def __init__(self, parent=None):
		super(ImageView, self).__init__(parent)
		self.cvImage = None
		self.scale = 1
		self.qtImage = None

	def set_image(self, img):
		self.cvImage = img
		self._update_scaled_image()

	def set_scale(self, scale):
		self.scale = scale
		self._update_scaled_image()

	def get_scale(self):
		return self.scale

	def paintEvent(self, QPaintEvent):
		painter = QtGui.QPainter()
		painter.begin(self)
		painter.drawImage(0, 0, self.qtImage)
		painter.end()

	def _update_scaled_image(self):
		if self.cvImage is None:
			self.qtImage = None
			return

		scaled_image = scale_image( self.cvImage, self.scale )
		if len( scaled_image.shape ) == 2:
			scaled_image = cv2.cvtColor(scaled_image, cv2.COLOR_GRAY2BGR)

		height, width, byteValue = scaled_image.shape
		byteValue = byteValue * width
		cv2.cvtColor(scaled_image, cv2.COLOR_BGR2RGB, scaled_image)

		self.qtImage = QtGui.QImage(scaled_image, width, height, byteValue, QtGui.QImage.Format_RGB888)
		self.resize( width, height )

class ImageWidget(QtGui.QWidget):
	u"""Widget для показа openCv изображения. Toolbar + ImageView"""
	def __init__(self, parent=None):
		super(ImageWidget, self).__init__(parent)

		zoomInAction = QtGui.QAction(QtGui.QIcon('resources/ZoomIn.png'), 'Zoom In', self)
		zoomInAction.triggered.connect(self.on_zoom_in)
		zoomOutAction = QtGui.QAction(QtGui.QIcon('resources/ZoomOut.png'), 'Zoom Out', self)
		zoomOutAction.triggered.connect(self.on_zoom_out)
		scale_label = QtGui.QLabel( self )
		scale_label.setText( "Scale:" )
		self.text_scale_field = QtGui.QLineEdit( self )
		self.text_scale_field.setText( str( 1 ) )
		self.text_scale_field.returnPressed.connect( self.on_scale_updated )
		
		toolbar = QtGui.QToolBar('Zoom', self)
		toolbar.addWidget( scale_label )
		toolbar.addWidget( self.text_scale_field )

		toolbar.addAction(zoomInAction)
		toolbar.addAction(zoomOutAction)

		self.image_view = ImageView( self )
		scroll_area = QtGui.QScrollArea( self )
		scroll_area.setWidget( self.image_view )

		self.setLayout( QtGui.QVBoxLayout( self ) )
		self.layout().addWidget( toolbar )
		self.layout().addWidget( scroll_area )

	def set_image( self, cvImage ):
		self.image_view.set_image( cvImage )

	def on_zoom_in( self ):
		new_scale = self.image_view.get_scale() + 0.1
		self._on_zoom( new_scale )

	def on_zoom_out( self ):
		new_scale = self.image_view.get_scale() - 0.1
		self._on_zoom( new_scale )

	def _on_zoom(self, new_zoom):
		if new_zoom <= 0.1 or new_zoom >= 10:
			return
		self.image_view.set_scale( new_zoom )
		self.text_scale_field.setText( str( self.image_view.get_scale() ) )

	def on_scale_updated( self ):
		newScale = 0
		try:
			newScale = float( self.text_scale_field.text().strip() )
		except Exception as e:
			newScale = self.image_view.get_scale()

		self.image_view.set_scale( newScale )

class DebugShowImageDialog(QtGui.QDialog):
	def __init__(self, image, parent=None):
		super(DebugShowImageDialog, self).__init__( parent )
		self.setLayout( QtGui.QVBoxLayout() )
		self.resize( 640, 480 )
		self.imageWidget = ImageWidget( self )
		self.imageWidget.set_image( image )
		self.layout().addWidget( self.imageWidget )

class DebugShowImageButton(QtGui.QPushButton):
	def __init__(self, cvImage, parent=None):
		super(DebugShowImageButton, self).__init__( parent )
		self.image = cvImage.copy()
		self.setText( "Show Image")
		self.clicked.connect( self._show_image )

	def _show_image(self):
		image_view = DebugShowImageDialog( self.image, self )
		image_view.exec_()

class GuiDebugHandler(QtGui.QScrollArea):
	def __init__(self, parent=None):
		super(GuiDebugHandler, self).__init__(parent)
		self.netsted_widget = QtGui.QWidget()
		self.netsted_widget.setLayout( QtGui.QVBoxLayout() )
		self.setWidget( self.netsted_widget )
		self.setWidgetResizable( True )

	def on_new_message(self, message, image):
		label = QtGui.QLabel( self )
		label.setText( message )
		self.netsted_widget.layout().addWidget( label )

		if image is not None:
			button = DebugShowImageButton( image, self )
			self.netsted_widget.layout().addWidget( button )

	def clear( self ):
		for i in reversed(range(self.netsted_widget.layout().count())):
			widget_to_remove = self.netsted_widget.layout().itemAt(i).widget()
			self.netsted_widget.layout().removeWidget( widget_to_remove )
			widget_to_remove.setParent( None )

class MainWindow(QtGui.QWidget):
	def __init__(self, parent=None):
		super(MainWindow, self).__init__(parent)

		self._init_gui()
		self._init_extractor()


	def _init_gui( self ):
		self.setLayout( QtGui.QVBoxLayout() )

		image_label = QtGui.QLabel( self )
		image_label.setText( "Please, enter image file path" )
		self.layout().addWidget( image_label )

		image_path_layout = QtGui.QHBoxLayout()
		self.image_path_edit = QtGui.QLineEdit( self )
		process_button = QtGui.QPushButton( self )
		process_button.setText( "Process" )
		process_button.clicked.connect( self.on_process )
		image_path_layout.addWidget( self.image_path_edit )
		image_path_layout.addWidget( process_button )
		self.layout().addLayout( image_path_layout )

		self.preprocessImag = QtGui.QCheckBox( self )
		self.preprocessImag.setText( "Use image preprocessing (WARNING! Long operation!)")
		self.layout().addWidget( self.preprocessImag )

		tabs = QtGui.QTabWidget()
		self.original_image = ImageWidget( self )
		self.processed_image = ImageWidget( self )
		self.debug_log = GuiDebugHandler( self )
		tabs.addTab( self.original_image, "ORIGINAL" )
		tabs.addTab( self.processed_image, "PROCESSED" )
		tabs.addTab( self.debug_log, "LOG" )

		LOG.set_handler( self.debug_log )
		LOG.set_enabled( True )

		self.layout().addWidget( tabs )


	def _init_extractor( self ):
		try:
			self.extractor = create_extractor()
		except Exception as e:
			self._excpetion_message_box( "Failed to create extractor in case of exception", e )
			sys.exit( -1 )


	def on_process( self ):
		try:
			LOG.clear()
			image = cv2.imread( self.image_path_edit.text() )
			if image is None:
				raise Exception( "Failed to open image \"" + self.image_path_edit.text() +"\"" )
			self.original_image.set_image( image )

			processed_image = self.extractor.process_image( image, self.preprocessImag.isChecked() )
			self.processed_image.set_image( processed_image )

		except Exception as e:
			self._excpetion_message_box( "Failed to process image in case of exception", e )


	def _excpetion_message_box( self, message, exception ):
		msg = QtGui.QMessageBox()
		msg.setIcon(QtGui.QMessageBox.Critical)
		msg.setText(message)
		msg.setInformativeText( str( exception ) )
		msg.setWindowTitle( "Error Occured" )
		msg.setStandardButtons( QtGui.QMessageBox.Ok )
		msg.exec_()