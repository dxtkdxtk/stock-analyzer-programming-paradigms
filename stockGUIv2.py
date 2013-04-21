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

def loadHistoricalData (tickerSymbol, dateStart, dateFinish):
	outputString = ystockquote.get_historical_prices(tickerSymbol, dateStart, dateFinish)
	return str(outputString).strip('[]')

class MainWindow(QMainWindow):
	def __init__(self, parent=None):
		super(MainWindow,self).__init__(parent)
		
		#Set the title and size of window
		self.setWindowTitle("Stock Analyzer")
		self.setGeometry(400,500,400,400)
		
		##FILE MENU##
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
		############
		
		#Create  a tab widget to manage GUI
		self.tab_widget=QTabWidget()
			
		#tab1 => Loading live data.
		#create all components of the tab
		self.button=QPushButton("Return Data:",self)
		self.textInfo=QLabel("Stock Ticker Symbol:")
		self.lineEdit=QLineEdit(self)
		self.textMatches=QLabel("Details:")
		self.matches=QTextBrowser(self)
		
		#create tab
		tab1=QWidget()		
		layout1 = QVBoxLayout(tab1)
		layout1.addWidget(self.textInfo)
		layout1.addWidget(self.lineEdit)
		layout1.addWidget(self.button)
		layout1.addWidget(self.textMatches)
		layout1.addWidget(self.matches)
		self.tab_widget.addTab(tab1, "Pull Live Data")
		
		#tab2 => Running analysis on data.
		#create componeonts of the tab
		self.dataLabel=QLabel("Data to Analyze:")
		self.dataToAnalyze=QTextBrowser(self)
		
		#create tab
		self.tab2=QWidget()
		layout2 = QVBoxLayout(self.tab2)
		layout2.addWidget(self.dataLabel)
		layout2.addWidget(self.dataToAnalyze)
		self.tab_widget.addTab(self.tab2, "Run Analysis")
		
		#tab3=> Pull historical data
		#create components
		self.stockTickerLabel = QLabel("Stock Ticker Label")
		self.stockTicker = QLineEdit(self)
		self.dateStartLabel = QLabel ("Start Date (\"YYYYMMDD\")")
		self.dateStart = QLineEdit(self)
		self.dateFinishLabel = QLabel ("Finish Date (\"YYYYMMDD\")")
		self.dateFinish = QLineEdit(self)
		self.historicalDataButton=QPushButton("Return Data", self)
		self.historicalDataLabel = QLabel("Historical Data")
		self.historicalData = QTextBrowser(self)
		self.cal = QCalendarWidget()
		
		#create tab
		tab3=QWidget()
		layout3= QVBoxLayout(tab3)
		layout3.addWidget(self.stockTickerLabel)
		layout3.addWidget(self.stockTicker)
		layout3.addWidget(self.dateStartLabel)
		layout3.addWidget(self.dateStart)
		layout3.addWidget(self.dateFinishLabel)
		layout3.addWidget(self.dateFinish)
		layout3.addWidget(self.historicalDataButton)
		layout3.addWidget(self.historicalDataLabel)
		layout3.addWidget(self.historicalData)
		layout3.addWidget(self.cal)
		self.tab_widget.addTab(tab3, "Historical Data Pull")

		#set up layout of overall GUI
		mainLayout=QVBoxLayout()
		mainLayout.addWidget(self.tab_widget)
		#create widget and use it to manipulate the layout of the different components
		widget=QWidget()
		widget.setLayout(mainLayout)
		self.setCentralWidget(widget)
	
	

		#manage connections of buttons
		self.connect(self.button,SIGNAL("clicked()"),self.buttonClick)
		self.connect(self.historicalDataButton,SIGNAL("clicked()"),self.historicalDataButtonClick)
		
		        # Connect the clicked signal to the centre handler
		self.connect(self.cal, SIGNAL('selectionChanged()'), self.date_changed)
		
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
				self.dataToAnalyze.setText(data)
			#change the focus to the second tab
			#tab2.show()
			self.tab_widget.setCurrentWidget (self.tab2)
			#tab2.raise_()
			
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
	
	def historicalDataButtonClick(self):
		#convery the inputs to text
		tickerSymbol = unicode(self.stockTicker.text())
		dateStart = unicode(self.dateStart.text())
		dateFinish = unicode (self.dateFinish.text())
		
		#check to make sure all have values given
		if (tickerSymbol == ''):
			self.historicalData.setText("No ticker symbol given!")
		if (dateStart == ''):
			self.historicalData.setText("No start date given!")
		if (dateFinish == ''):
			self.historicalData.setText("No finish date given!")
		outputString = loadHistoricalData(tickerSymbol, dateStart, dateFinish)
		self.historicalData.setText(outputString)
		
		
	def date_changed(self):
		# Fetch the currently selected date, this is a QDate object
		date = self.cal.selectedDate()
		# This is a gives us the date contained in the QDate as a native
		# python date[time] object
		pydate = date.toPyDate()
		# Show this date in our label
		self.historicalData.setText('The date is: %s' % pydate)
		
#set up and begin the application
app=QApplication(sys.argv)
window=MainWindow()
window.show()
app.exec_()
