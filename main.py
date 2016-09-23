import sys
from gui import MainWindow
from PyQt4 import QtGui

def main():
	app = QtGui.QApplication(sys.argv)
	ex = MainWindow()
	ex.resize( 640, 480 )
	ex.show()
	sys.exit(app.exec_())

main()