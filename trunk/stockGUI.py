#!/usr/bin/env python
#Programming Paradigms
#Final Project

from PyQt4.QtCore import *
from PyQt4.QtGui import *
import sys

import ystockquote
import pyqtgraph as pg
import numpy as np
import SACudaProxy



##Helper Functions
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

def convertDate(dateValue):
		dateString = dateValue.toPyDate()
		yearString=str(dateString.year)
		if (dateString.month < 10):
			monthString="0"+str(dateString.month)
		else:
			monthString=str(dateString.month)
		if (dateString.day < 10):
			dayString="0"+str(dateString.day)
		else:
			dayString=str(dateString.day)
			
		dateStringFinal=yearString+monthString+dayString
		return dateStringFinal

def loadHistoricalData (tickerSymbol, dateStart, dateFinish):
	outputString = ystockquote.get_historical_prices(tickerSymbol, dateStart, dateFinish)
	return str(outputString).strip('[]')

class MainWindow(QMainWindow):
	def __init__(self, parent=None):
		super(MainWindow,self).__init__(parent)
		
		#Set the title and size of window
		self.setWindowTitle("Stock Analyzer")
		self.setGeometry(0,0,1000,675)
		
		### FILE MENU ###
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

		### TABS ###
		#Create  a tab widget to manage GUI
		self.tab_widget=QTabWidget()
			
		#tab1 => Loading live data.
		#create all components of the tab
		self.currentButton=QPushButton("Return Data:",self)
		self.textInfo=QLabel("Stock Ticker Symbol:")
		self.lineEdit=QLineEdit(self)
		self.textMatches=QLabel("Details:")
		self.matches=QTextBrowser(self)
		#create tab
		tab1=QWidget()		
		layout1 = QVBoxLayout(tab1)
		layout1.addWidget(self.textInfo)
		layout1.addWidget(self.lineEdit)
		layout1.addWidget(self.currentButton)
		layout1.addWidget(self.textMatches)
		layout1.addWidget(self.matches)
		self.tab_widget.addTab(tab1, "Pull Live Data")
				
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
		self.tickerOneLabel = QLabel("Stock Ticker (\"blue +\"):")
		self.tickerOne = QLineEdit(self)
		self.tickerTwoLabel = QLabel("Stock Ticker (\"green o\"):")
		self.tickerTwo = QLineEdit(self)
		self.compareStartLabel = QLabel("Start Date")
		self.compareStart = QCalendarWidget()
		self.compareFinishLabel = QLabel("Finish Date")
		self.compareFinish = QCalendarWidget()
		self.compareButton = QPushButton("Compare!", self)
		self.saveCompareButton = QPushButton("Save!", self)	
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
		layout5.addWidget(self.saveCompareButton)
		self.tab_widget.addTab(tab5,"Comparison")
		
		#tab6 => comparison graph
		#create components
		self.compareGraph=pg.PlotWidget()
		#create tab
		tab6=QWidget()
		layout6=QVBoxLayout(tab6)
		layout6.addWidget(self.compareGraph)
		self.tab_widget.addTab(tab6,"Comparison Graph")		
		
		#tap9 => inverse analsis graph
		#out of order but alot of work to rename #s
		#create components
		self.inverseGraph=pg.PlotWidget()
		#create tab
		tab9=QWidget()
		layout9=QVBoxLayout(tab9)
		layout9.addWidget(self.inverseGraph)
		self.tab_widget.addTab(tab9,"Inverse Analysis Graph")		
		
		#tab7 => market average tab
		#create components
		fTickers=open('NYSE.ticker')
		lines = fTickers.readlines()
		self.MAStock=QTableWidget(len(lines),1)
		i=0
		for item in lines:
			var=QTableWidgetItem(item.strip("\r\n"))
			var.setFlags(Qt.ItemIsSelectable|Qt.ItemIsEnabled)
			self.MAStock.setItem(i,0,var)
			i=i+1
		self.MAVarLabel = QLabel ("Variable to Analyze:")
		self.MAVar= QComboBox()
		#add values to combo box
		MAVariables = ("Price","Volume")
		self.MAVar.addItems(MAVariables)
		self.MAStartLabel = QLabel("Start Date")
		self.MAStart = QCalendarWidget()
		self.MAFinishLabel = QLabel("Finish Date")
		self.MAFinish = QCalendarWidget()
		self.MAButton = QPushButton("Market Average", self)
		#create tab
		tab7=QWidget()
		layout7=QVBoxLayout(tab7)
		layout7.addWidget(self.MAStock)
		layout7.addWidget(self.MAVarLabel)
		layout7.addWidget(self.MAVar)
		layout7.addWidget(self.MAStartLabel)
		layout7.addWidget(self.MAStart)
		layout7.addWidget(self.MAFinishLabel)
		layout7.addWidget(self.MAFinish)
		layout7.addWidget(self.MAButton)
		self.tab_widget.addTab(tab7,"Market Average")
		
		#tab8 => market average graph
		#create components
		self.MAGraph=pg.PlotWidget()
		#create tab
		tab8=QWidget()
		layout8=QVBoxLayout(tab8)
		layout8.addWidget(self.MAGraph)
		self.tab_widget.addTab(tab8,"Market Average Graph")

		#tab2 => Running analysis on data.
		#create componeonts of the tab
		self.dataLabel=QLabel("Loaded Data View:")
		self.dataToAnalyze=QTextBrowser(self)
		self.inverseAnalysisCUDAButton = QPushButton("CUDA - Inverse Analysis", self)
		self.marketAnalysisCUDAButton = QPushButton("CUDA - Market Analysis", self)
		#create tab
		self.tab2=QWidget()
		layout2 = QVBoxLayout(self.tab2)
		layout2.addWidget(self.dataLabel)
		layout2.addWidget(self.dataToAnalyze)
		layout2.addWidget(self.inverseAnalysisCUDAButton)
		layout2.addWidget(self.marketAnalysisCUDAButton)
		self.tab_widget.addTab(self.tab2, "Loaded Data")
		
		### LAYOUT ###
		#set up layout of overall GUI
		mainLayout=QVBoxLayout()
		mainLayout.addWidget(self.tab_widget)
		#create widget and use it to manipulate the layout of the different components
		widget=QWidget()
		widget.setLayout(mainLayout)
		self.setCentralWidget(widget)
	
		### CONNECTIONS ###
		# Manage connections of buttons
		self.connect(self.currentButton,SIGNAL("clicked()"),self.currentButtonClick)
		self.connect(self.historicalDataButton,SIGNAL("clicked()"),self.historicalDataButtonClick)
		self.connect(self.compareButton,SIGNAL("clicked()"),self.compareButtonClick)
		self.connect(self.inverseAnalysisCUDAButton,SIGNAL("clicked()"),self.inverseAnalysisCUDA)
		self.connect(self.MAButton,SIGNAL("clicked()"),self.marketAnalysis)
		self.connect(self.marketAnalysisCUDAButton,SIGNAL("clicked()"),self.marketAnalysisCUDA)
		self.connect(self.saveCompareButton,SIGNAL("clicked()"),self.inverseAnalysis)
		# Manage calendar changes
		self.connect(self.calStart, SIGNAL('selectionChanged()'), self.date_changed)
		self.connect(self.calFinish, SIGNAL('selectionChanged()'), self.date_changed)


	##GUI FUNCTIONS##
		
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
			self.tab_widget.setCurrentWidget (self.tab2)

	#save current Data		
	def fileSaveCurrent(self):
		FilePath = QFileDialog.getSaveFileName()
		if(FilePath):
			f=open(FilePath,'w')
			f.write(str(self.matches.toPlainText()))
	
	#save historical Data
	def fileSaveHistorical(self):
		FilePath = QFileDialog.getSaveFileName()
		if(FilePath):
			f=open(FilePath,'w')
			f.write(str(self.historicalData.toPlainText()))
	
	#function to manage the button click parse functionality
	def currentButtonClick(self):
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
		dateStartString=convertDate(dateStart)
		dateFinishString=convertDate(dateFinish)
		
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
		#only display symbols is length is less than a certain value
		if (len(y)<15):
				self.graph.plot(y,pen='b',symbol='+')
		else:
				self.graph.plot(y,pen='b')

				
	def compareButtonClick(self):
		#convery the inputs to text
		tickerSymbolOne = unicode(self.tickerOne.text())
		tickerSymbolTwo = unicode(self.tickerTwo.text())
	
		#pull dates from calendar	
		dateStart = self.compareStart.selectedDate()
		dateFinish = self.compareFinish.selectedDate()	
		#convert dates to correct format for ystockquote
		dateStartString=convertDate(dateStart)
		dateFinishString=convertDate(dateFinish)
		
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
		if (len(y)<15):
				self.compareGraph.plot(y,pen='b',symbol='+')
		else:
				self.compareGraph.plot(y,pen='b')		
		
		#graph second output
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
		
		if (len(y)<15):
				self.compareGraph.plot(y,pen='g',symbol='o')
		else:
				self.compareGraph.plot(y,pen='g')				


	def inverseAnalysis(self):
		#convery the inputs to text
		tickerSymbolOne = unicode(self.tickerOne.text())
		tickerSymbolTwo = unicode(self.tickerTwo.text())
	
		#pull dates from calendar	
		dateStart = self.compareStart.selectedDate()
		dateFinish = self.compareFinish.selectedDate()	
		#convert dates to correct format for ystockquote
		dateStartString=convertDate(dateStart)
		dateFinishString=convertDate(dateFinish)
		
		#load data
		outputString1 = loadHistoricalData(tickerSymbolOne, dateStartString, dateFinishString)
		outputString2 = loadHistoricalData(tickerSymbolTwo, dateStartString, dateFinishString)
		
		#parse first stock data to make a list of prices
		oString1=""		
		for item in outputString1.split("], ["):
			oString1+=item
			oString1+="\n"		
		i=0
		price1=[]
		for line in oString1.split("\n"):
			if ((i!=0) and (i!=(len(oString1.split("\n"))-1))):
				item =line.split(",")
				price1.append(float(item[4].strip('\"\' ')))
			i=i+1
				
		#parse second stock data to make a list of prices
		oString2=""		
		for item in outputString2.split("], ["):
			oString2+=item
			oString2+="\n"				
		i=0
		price2=[]
		for line in oString2.split("\n"):
			if ((i!=0) and (i!=(len(oString2.split("\n"))-1))):
				item =line.split(",")
				price2.append(float(item[4].strip('\"\' ')))
			i=i+1
			
		#create a string with prices for each date side by side
		finalString=""
		i=0
		for item in price1:
				finalString+="("
				finalString+=str(item)
				finalString+=","
				finalString+=str(price2[i])
				finalString+=")"
				if (i != len(price1)-1):
						finalString+="\n"
				i=i+1
				
		#open a file for user to save data
		FilePath = QFileDialog.getSaveFileName()
		if(FilePath):
			f=open(FilePath,'w')
			f.write(finalString)
		


	def inverseAnalysisCUDA(self):		
		#pull data from text field
		dataString = self.dataToAnalyze.toPlainText()
		dataString = str(dataString)
		#parse into list of lists
		dataList=[]
		for line in dataString.split("\n"):
				tempList=[]
				for item in line.strip("\n ").split(","):
					tempList.append(float(item))
					#map(float,tempList)
				dataList.append(tempList)

		#pass to CUDA
		handle = SACudaProxy.SACudaProxy()
		dataReturn = handle.FindInverseTrends(dataList)
		
		#graph the return data
		self.inverseGraph.clear()
		
		#create list of prices for each stock
		stockOneList=[]
		stockTwoList=[]
		for line in dataList:
			stockOneList.append(line[0])
			stockTwoList.append(line[1])
		
		#plot stocks
		self.inverseGraph.plot(stockOneList,pen='b',symbol='x')
		self.inverseGraph.plot(stockTwoList,pen='g',symbol='o')
		
		#plot markers based on CUDA call
		for item in dataReturn:
			avg = (stockOneList[item]+stockTwoList[item])/2
			self.graph.plot([item-1,item,item+1],[avg,avg,avg],pen='r',symbol="o")
		#SHOULD prob change symbol to dif symbol than stocks.. forget what the others are

	def marketAnalysis(self):
		#pull dates from calendar	
		dateStart = self.MAStart.selectedDate()
		dateFinish = self.MAFinish.selectedDate()	
		#convert dates to correct format for ystockquote
		dateStartString=convertDate(dateStart)
		dateFinishString=convertDate(dateFinish)

		#convert the selected stocks into a list of text stocks
		stockList=[]
		for stock in self.MAStock.selectedItems():
			stockList.append(str(stock.text()))
		
		#create the list of lists to pass to CUDA
		outputList=[]
		for stock in stockList:
			#load data for each ticker
			data = loadHistoricalData(stock, dateStartString, dateFinishString)
			#skip invalid tickers
			tempList=[]
			if (data[0:12]=="'<!doctype h"):
				pass
			else:
				#remove [] and create new lines between each date
				tempString=""
				for line in data.split("], ["):
					tempString+=line
					tempString+="\n"
				i=0
				for line in tempString.split("\n"):
					if ((i!=0) and (i!=(len(tempString.split("\n"))-1))):
						item=line.split(",")
						#check whether price or volume
						if (self.MAVar.currentText()=="Price"):
							tempList.append(float(item[4].strip('\"\' ')))
						if (self.MAVar.currentText()=="Volume"):
							tempList.append(float(item[5].strip('\"\' ')))
					i=i+1
				outputList.append(tempList)	
		
		#convert to a string
		dataString=""
		j=0
		for item in outputList:
			i=0
			for data in item:
				if (i!=len(item)-1):
					dataString+=str(data)
					dataString+=","
				else:
					dataString+=str(data)
				i=i+1
			j=j+1
			if (j!=len(outputList)):
				dataString+="\n"

		#open option to save string in file
		FilePath = QFileDialog.getSaveFileName()
		if(FilePath):
			f=open(FilePath,'w')
			f.write(dataString)
			

	def marketAnalysisCUDA(self):		
		#pull data from field
		dataString = self.dataToAnalyze.toPlainText()
		dataString = str(dataString)
		#convert to list of lists
		dataList=[]
		checkLength=len(dataString.split("\n")[0])
		for line in dataString.split("\n"):
				#make sure all stocks have existed for the same amount of time
				if (len(line)==checkLenth):
					tempList=[]
					for item in line.strip("\n ").split(","):
						tempList.append(float(item))
						#map(float,tempList)
					dataList.append(tempList)
			
		#pass to CUDA
		handle = SACudaProxy.SACudaProxy()
		dataReturn = handle.CalculateMarketAverage(dataList)
		
		#graph results
		self.MAGraph.clear()
		if (len(dataReturn)<15):
				self.MAGraph.plot(dataReturn,pen='b',symbol='x')
		else:
				self.MAGraph.plot(dataReturn,pen='b')		
	
	
	def date_changed(self):
		# Indicate to the user that the date has changed
		self.historicalData.setText("The date has changed! Press button to refresh")


		
		
#set up and begin the application
app=QApplication(sys.argv)
window=MainWindow()
window.show()
app.exec_()

