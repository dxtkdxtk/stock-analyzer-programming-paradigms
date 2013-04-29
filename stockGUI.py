#!/usr/bin/env python
#Programming Paradigms
#Final Project

from PyQt4.QtCore import *
from PyQt4.QtGui import *
import sys

import ystockquote
import pyqtgraph as pg
import numpy as np

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
		self.setGeometry(0,0,650,550)
		
		##FILE MENU##
		#create the action for opening a file
		loadAction = QAction('Load',self)        
		loadAction.setStatusTip('Load a file')
		loadAction.triggered.connect(self.fileLoad)
		
		#create the action for saving a file of the current data
		saveActionCurrent = QAction('Save Current',self)
		saveActionCurrent.setStatusTip('Save a file containing the current data')
		saveActionCurrent.triggered.connect(self.fileSaveCurrent)
		
		#create the action for saving a file of the historical data
		saveActionHistorical = QAction('Save Historical',self)
		saveActionHistorical.setStatusTip('Save a file containing the historical data')
		saveActionHistorical.triggered.connect(self.fileSaveHistorical)
		
		#displays status tip (for the open file selection)
		self.statusBar()
				
		#create a menu bar with file as a choice and open as one of the actions under it		
		menubar = self.menuBar()
		fileMenu = menubar.addMenu('&File')
		fileMenu.addAction(loadAction)
		fileMenu.addAction(saveActionCurrent)
		fileMenu.addAction(saveActionHistorical)
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
		self.dateStartLabel = QLabel ("Start Date")
		self.calStart = QCalendarWidget()
		self.dateFinishLabel = QLabel ("Finish Date")
		self.calFinish = QCalendarWidget()
		self.historicalDataButton=QPushButton("Return Data", self)
		self.historicalDataLabel = QLabel("Historical Data")
		self.historicalData = QTextBrowser(self)

		#create tab
		tab3=QWidget()
		layout3= QVBoxLayout(tab3)
		layout3.addWidget(self.stockTickerLabel)
		layout3.addWidget(self.stockTicker)
		layout3.addWidget(self.dateStartLabel)
		layout3.addWidget(self.calStart)
		layout3.addWidget(self.dateFinishLabel)
		layout3.addWidget(self.calFinish)
		layout3.addWidget(self.historicalDataButton)
		layout3.addWidget(self.historicalDataLabel)
		layout3.addWidget(self.historicalData)
		self.tab_widget.addTab(tab3, "Historical Data Pull")
		

		#tab4 =>historical graphs
		#create components
		self.graphLabel= QLabel("Historical Graph:")
		self.graph=pg.PlotWidget()
		
		#create tab
		tab4=QWidget()
		layout4= QVBoxLayout(tab4)
		layout4.addWidget(self.graphLabel)
		layout4.addWidget(self.graph)
		self.tab_widget.addTab(tab4, "Historical Graph")
		
		
		#tab5 => comparison
		#create components
		self.tickerOneLabel = QLabel("Stock Ticker (\"o\"):")
		self.tickerOne = QLineEdit(self)
		self.tickerTwoLabel = QLabel("Stock Ticker (\"+\"):")
		self.tickerTwo = QLineEdit(self)
		self.compareStartLabel = QLabel("Start Date")
		self.compareStart = QCalendarWidget()
		self.compareFinishLabel = QLabel("Finish Date")
		self.compareFinish = QCalendarWidget()
		self.compareButton = QPushButton("Compare!", self)
		
		#create tab
		tab5=QWidget()
		layout5=QVBoxLayout(tab5)
		layout5.addWidget(self.tickerOneLabel)
		layout5.addWidget(self.tickerOne)
		layout5.addWidget(self.tickerTwoLabel)
		layout5.addWidget(self.tickerTwo)
		layout5.addWidget(self.compareStartLabel)
		layout5.addWidget(self.compareStart)
		layout5.addWidget(self.compareFinishLabel)
		layout5.addWidget(self.compareFinish)
		layout5.addWidget(self.compareButton)

		self.tab_widget.addTab(tab5,"Comparison")
		
		
		#tab6 => comparison graph
		#create components
		self.compareGraph=pg.PlotWidget()
		
		#create tab
		tab6=QWidget()
		layout6=QVBoxLayout(tab6)
		layout6.addWidget(self.compareGraph)
		self.tab_widget.addTab(tab6,"Comparison Graph")

		#set up layout of overall GUI
		mainLayout=QVBoxLayout()
		mainLayout.addWidget(self.tab_widget)
		#create widget and use it to manipulate the layout of the different components
		widget=QWidget()
		widget.setLayout(mainLayout)
		self.setCentralWidget(widget)
	
	

		# Manage connections of buttons
		self.connect(self.button,SIGNAL("clicked()"),self.buttonClick)
		self.connect(self.historicalDataButton,SIGNAL("clicked()"),self.historicalDataButtonClick)
		self.connect(self.compareButton,SIGNAL("clicked()"),self.compareButtonClick)
		# Manage calendar changes
		self.connect(self.calStart, SIGNAL('selectionChanged()'), self.date_changed)
		self.connect(self.calFinish, SIGNAL('selectionChanged()'), self.date_changed)
		
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
			
	def fileSaveCurrent(self):
		FilePath = QFileDialog.getSaveFileName()
		if(FilePath):
			f=open(FilePath,'w')
			f.write(str(self.matches.toPlainText()))
	
	def fileSaveHistorical(self):
		FilePath = QFileDialog.getSaveFileName()
		if(FilePath):
			f=open(FilePath,'w')
			f.write(str(self.historicalData.toPlainText()))
	
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
	
		#pull dates from calendar	
		dateStart = self.calStart.selectedDate()
		dateFinish = self.calFinish.selectedDate()
		
		#convert dates to correct format for ystockquote
		#start date
		dateS = dateStart.toPyDate()
		sYearString=str(dateS.year)
		if (dateS.month < 10):
			sMonthString="0"+str(dateS.month)
		else:
			sMonthString=str(dateS.month)
		if (dateS.day < 10):
			sDayString="0"+str(dateS.day)
		else:
			sDayString=str(dateS.day)
			
		dateStartString=sYearString+sMonthString+sDayString
		
		#finish date
		dateF = dateFinish.toPyDate()
		fYearString=str(dateF.year)
		if (dateF.month < 10):
			fMonthString="0"+str(dateF.month)
		else:
			fMonthString=str(dateF.month)
		if (dateF.day < 10):
			fDayString="0"+str(dateF.day)
		else:
			fDayString=str(dateF.day)
			
		dateFinishString=fYearString+fMonthString+fDayString
		
		#set output
		outputString = loadHistoricalData(tickerSymbol, dateStartString, dateFinishString)
		oString=""
		for item in outputString.split("], ["):
			oString+=item
			oString+="\n"
		self.historicalData.setText(oString)

		#test if an error occured and display to the user if so
		if (outputString[0:12]=="'<!doctype h"):
			self.historicalData.setText("Invalid Ticker or invalid date!")
	
		#check to make sure ticker value given
		if (tickerSymbol == ''):
			self.historicalData.setText("No ticker symbol given!")
		
		
		#update graph
		x=[]
		y=[]
		i=0
		for line in oString.split("\n"):
			if ((i!=0) and (i!=(len(oString.split("\n"))-1))):
				item =line.split(",")
				x.append(item[0].strip('\"\' -'))
				y.append(float(item[4].strip('\"\' ')))
			i=i+1
		self.graph.clear()
		self.graph.plot(y,pen='b',symbol='+')
		
	def compareButtonClick(self):
		#convery the inputs to text
		tickerSymbolOne = unicode(self.tickerOne.text())
		tickerSymbolTwo = unicode(self.tickerTwo.text())
	
		#pull dates from calendar	
		dateStart = self.compareStart.selectedDate()
		dateFinish = self.compareFinish.selectedDate()
		
		#convert dates to correct format for ystockquote
		#start date
		dateS = dateStart.toPyDate()
		sYearString=str(dateS.year)
		if (dateS.month < 10):
			sMonthString="0"+str(dateS.month)
		else:
			sMonthString=str(dateS.month)
		if (dateS.day < 10):
			sDayString="0"+str(dateS.day)
		else:
			sDayString=str(dateS.day)
			
		dateStartString=sYearString+sMonthString+sDayString
		
		#finish date
		dateF = dateFinish.toPyDate()
		fYearString=str(dateF.year)
		if (dateF.month < 10):
			fMonthString="0"+str(dateF.month)
		else:
			fMonthString=str(dateF.month)
		if (dateF.day < 10):
			fDayString="0"+str(dateF.day)
		else:
			fDayString=str(dateF.day)
			
		dateFinishString=fYearString+fMonthString+fDayString
		
		#graph first output
		outputString = loadHistoricalData(tickerSymbolOne, dateStartString, dateFinishString)
		oString=""
		for item in outputString.split("], ["):
			oString+=item
			oString+="\n"		
		
		#update graph
		x=[]
		y=[]
		i=0
		for line in oString.split("\n"):
			if ((i!=0) and (i!=(len(oString.split("\n"))-1))):
				item =line.split(",")
				x.append(item[0].strip('\"\' -'))
				y.append(float(item[4].strip('\"\' ')))
			i=i+1
		
		self.compareGraph.clear()
		self.compareGraph.plot(y,pen='b',symbol='+')
		
		
		#graoh second output
		outputString = loadHistoricalData(tickerSymbolTwo, dateStartString, dateFinishString)
		oString=""
		for item in outputString.split("], ["):
			oString+=item
			oString+="\n"		
		
		#update graph
		x=[]
		y=[]
		i=0
		for line in oString.split("\n"):
			if ((i!=0) and (i!=(len(oString.split("\n"))-1))):
				item =line.split(",")
				x.append(item[0].strip('\"\' -'))
				y.append(float(item[4].strip('\"\' ')))
			i=i+1
		
		self.compareGraph.plot(y,pen='g',symbol='o')
		
	def date_changed(self):
		# Indicate to the user that the date has changed
		self.historicalData.setText("The date has changed! Press button to refresh")
		
#set up and begin the application
app=QApplication(sys.argv)
window=MainWindow()
window.show()
app.exec_()