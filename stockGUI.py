#!/usr/bin/env python
#Programming Paradigms
#Final Project

from PyQt4.QtCore import *
from PyQt4.QtGui import *
import sys

import ystockquote

def loadData(tickerSymbol):
	price=ystockquote.get_price(tickerSymbol)
	volume=ystockquote.get_volume(tickerSymbol)
	marketCap=ystockquote.get_market_cap(tickerSymbol)
	bookValue=ystockquote.get_book_value(tickerSymbol)
	dividendPerShare=ystockquote.get_dividend_per_share(tickerSymbol)
	earningsPerShare=ystockquote.get_earnings_per_share(tickerSymbol)
	priceEarningsRatio=ystockquote.get_price_earnings_ratio(tickerSymbol)
	shortRatio=ystockquote.get_short_ratio(tickerSymbol)
	
	outputString=tickerSymbol.upper()+':\n'
	outputString+="Price: $%s\n" % price
	outputString+="Volume: %s\n" % volume
	outputString+="Market Cap: %s\n" % marketCap
	outputString+="Book Value: %s\n" % bookValue
	outputString+="Dividend per Share: %s\n" % dividendPerShare
	outputString+="Earnings per Share: %s\n" % earningsPerShare
	outputString+="Price Earnings Ratio: %s\n" % priceEarningsRatio
	outputString+="Short Ratio: %s\n" % shortRatio
	return outputString

class MainWindow(QMainWindow):
	def __init__(self, parent=None):
		super(MainWindow,self).__init__(parent)
		
		#create the action for opening a file
		loadAction = QAction('Load',self)        
		loadAction.setStatusTip('Load a file')
		loadAction.triggered.connect(self.fileLoad)
		
		#displays status tip (for the open file selection)
		self.statusBar()
		
		#create a menu bar with file as a choice and open as one of the actions under it		
		menubar = self.menuBar()
		fileMenu = menubar.addMenu('&File')
		fileMenu.addAction(loadAction)

		#create all other components of the GUI
		self.button=QPushButton("Go!", self)
		self.textInfo=QLabel("Stock Ticker Symbol:")
		self.lineEdit=QLineEdit(self)
		self.textMatches=QLabel("Details:")
		self.matches=QTextBrowser(self)
		
		#set the title
		self.setWindowTitle("Stock Analyzer")
		#set the size of the window
		self.setGeometry(300,400,300,300)
		
		#create widget and use it to manipulate the layout of the different components
		widget=QWidget()
		layout = QVBoxLayout()
		layout.addWidget(self.textInfo)
		layout.addWidget(self.lineEdit)
		layout.addWidget(self.button)
		layout.addWidget(self.textMatches)
		layout.addWidget(self.matches)
		widget.setLayout(layout)
		self.setCentralWidget(widget)
		
		#manage connections
		self.connect(self.button,SIGNAL("clicked()"),self.buttonClick)
		
	#function to manage opening files	
	def fileLoad(self):
		#open a QFileDialog to get the file name
		fname = QFileDialog.getOpenFileName(self, 'Open file','//')
		#checks to make sure the user didnt click cancel
		if fname:
			#open the file and read it into the textEdit
			f = open(fname, 'r')
			with f:        
				data = f.read()
				self.textEdit.setText(data)
				
	#function to manage the button click parse functionality
	def buttonClick(self):
		#convert the inputs to text
		tickerSymbol = unicode(self.lineEdit.text())
		
		#check text or both are empty (domain alone is checked w/ the exception as this domain will through an exception)
		if (tickerSymbol ==''):
			self.matches.setText("No ticker symbol given!")
		#if valid continue and use regular expressions to search for emails		
		else:
			#check to ensure ticker is valid (if stock exchange is n/a it is not valid)
			if (ystockquote.get_stock_exchange(tickerSymbol)=='N/A'):
				self.matches.setText('Invalid Ticker')
			else:
				outputString=loadData(tickerSymbol)				
				self.matches.setText(outputString)


#set up and begin the application
app=QApplication(sys.argv)
window=MainWindow()
window.show()
app.exec_()
