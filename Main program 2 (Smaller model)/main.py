########################################################################
## IMPORTS
########################################################################
import os, shutil
import time
import mysql.connector
import sys
from PyQt5 import sip
import requests
import cv2
import numpy as np
import imutils
from PyQt5.QtCore import QThread,pyqtSignal
from PIL import Image 
import torchvision.models as models
import torch.nn as nn
import torch
import cv2
import torchvision.transforms as transforms
import argparse
from PyQt5.QtCore import QRunnable, Qt, QThreadPool,QMutex
import webbrowser
########################################################################
# IMPORT GUI FILE
from test import *
########################################################################

########################################################################
# IMPORT Custom widgets
from Custom_Widgets.Widgets import *
# INITIALIZE APP SETTINGS
settings = QSettings()

#Import loading gif class
from waitingspinnerwidget import QtWaitingSpinner

#Import runtime functions class
from RuntimeFunctions import *
from ParallelThreadFunctions import *
########################################################################

#Global
Productsetupcounter=0



########################################################################
## MAIN WINDOW CLASS
########################################################################

#SIGNALS CLASS

                   
                          
                            






class MainWindow(QMainWindow):
    
    def gotoimage(self,currentrow):
        global Productsetupcounter
        Productsetupcounter=currentrow
        self.ui.SetupModel.click()
        RuntimeFunctions.Setupproductiamge(self,Productsetupcounter)

 
    
    def gotonextimage(self):
        global Productsetupcounter


        if(Productsetupcounter==601):
            return 
        
        productinfo=files[Productsetupcounter].split(".")
        productinfo2=productinfo[0].split("-")
        productinfo2
        
        (self.ui.SkuID.text()!=productinfo2[4])
        os.rename("ModelInfo\\imageforkaim\\"+files[Productsetupcounter], "ModelInfo\\imageforkaim\\" + productinfo2[0]+ "-" + productinfo2[1] + "-" + productinfo2[2] + "-" + productinfo2[3] + "-" + self.ui.SkuID.text() + ".jpg")
        files[Productsetupcounter]=productinfo2[0]+ "-" + productinfo2[1] + "-" + productinfo2[2] + "-" + productinfo2[3] + "-" + self.ui.SkuID.text() + ".jpg"
        
        Productsetupcounter+=1
        RuntimeFunctions.Setupproductiamge(self,Productsetupcounter)
        
    def gotopreviousimage(self):
        global Productsetupcounter
        if(Productsetupcounter==0):
            return
        
        productinfo=files[Productsetupcounter].split(".")
        productinfo2=productinfo[0].split("-")
        productinfo2
        
        (self.ui.SkuID.text()!=productinfo2[4])
        os.rename("ModelInfo\\imageforkaim\\"+files[Productsetupcounter], "ModelInfo\\imageforkaim\\" + productinfo2[0]+ "-" + productinfo2[1] + "-" + productinfo2[2] + "-" + productinfo2[3] + "-" + self.ui.SkuID.text() + ".jpg")
        files[Productsetupcounter]=productinfo2[0]+ "-" + productinfo2[1] + "-" + productinfo2[2] + "-" + productinfo2[3] + "-" + self.ui.SkuID.text() + ".jpg"
        
        Productsetupcounter-=1
        RuntimeFunctions.Setupproductiamge(self,Productsetupcounter)        
        
    
    
    def __init__(self, parent=None):
        QMainWindow.__init__(self)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        ########################################################################
        # APPLY JSON STYLESHEET
        ########################################################################
        # self = QMainWindow class
        # self.ui = Ui_MainWindow / user interface class
        #Use this if you only have one json file named "style.json" inside the root directory, "json" directory or "jsonstyles" folder.
        loadJsonStyle(self, self.ui) 

        # Use this to specify your json file(s) path/name
        # loadJsonStyle(self, self.ui, jsonFiles = {
        #     "mystyle.json", "style.json"
        #     }) 

        ########################################################################

        #Pre launch code

        # setting font to the style sheet
        self.ui.WebcamRadio.setStyleSheet("QRadioButton"
                                        "{"
                                        "font : 8px Arial;"
                                        "}")
        self.ui.WebcamRadio.adjustSize()

        self.ui.IpWebcamRadio.setStyleSheet("QRadioButton"
                                        "{"
                                        "font : 8px Arial;"
                                        "}")
        self.ui.IpWebcamRadio.adjustSize()


        
        self.ui.NotificationNumber.setVisible(False)
        self.ui.SkuID.setEnabled(False)
        #This will setup the database if it had already been saved before
        
        #Read settings file
        with open('settings\setting.txt', 'r') as f:
            settings = f.readlines()    
              
             #If the settings file contains any text set up the database from the text file
            if len(settings)!=0:
                ParallelThreadFunctions.SetupDatabase(self,0,0,True)
        
        
        #######################################################################
        # SHOW WINDOW
        #######################################################################
        self.show() 

        #Run the setupdatabase function when the user clicks the setup database button.
        self.ui.SetupDatabase.clicked.connect(lambda : ParallelThreadFunctions.SetupDatabase(self,0,0,False))
        

        
        #Refresh the database
        self.ui.Refreshdatabase.clicked.connect(lambda : ParallelThreadFunctions.RefreshDatabase(self))
        
        #Refresh the database
        self.ui.Savechanges.clicked.connect(lambda : ParallelThreadFunctions.Savechanges(self))   
        
        #Setup up model button
        self.ui.SetupModel.clicked.connect(lambda : RuntimeFunctions.Setupproductiamge(self,Productsetupcounter)) 
        
        #If model table cell is double clicked
        self.ui.modeltable.doubleClicked.connect(lambda : MainWindow.gotoimage(self,self.ui.modeltable.currentRow()))
        
        #Go to next image in model setup
        self.ui.ModelNext.clicked.connect(lambda : MainWindow.gotonextimage(self)) 
        
        #Go to previous image in model setup
        self.ui.ModelPrevious.clicked.connect(lambda : MainWindow.gotopreviousimage(self))    
        
        #If modelViewer is clicked
        self.ui.ModelViewer.clicked.connect(lambda : RuntimeFunctions.SetupModelTable(self))  
        
        #If Cameratab is clicked
        self.ui.Cameras.clicked.connect(lambda : RuntimeFunctions.UpdateCameraTable(self))  
          
        #If Add camera is clicked   
        self.ui.SetupCamera.clicked.connect(lambda : ParallelThreadFunctions.AddCamera(self))  
        
        #When a cell in the camera table is clicked call this function 
        self.ui.CameraTable.cellClicked.connect(lambda : RuntimeFunctions.getClickedCell(self,self.ui.CameraTable.currentRow()))
        
        #When the view footage button is clicked
        self.ui.ViewFootage.clicked.connect(lambda : ParallelThreadFunctions.ViewFootage(self))  
        
        #When the delete record button in the camera tab is clicked
        self.ui.DeleterecordCamera.clicked.connect(lambda : RuntimeFunctions.DeleteCameraRecord(self))  
        
           
        #When the Refresh button in the camera tab is clicked
        self.ui.RefreshCameraTable.clicked.connect(lambda : RuntimeFunctions.RefreshCameraTable(self))       
        
              
        #When the save button in the camera tab is clicked
        self.ui.SavechangesCamera.clicked.connect(lambda : RuntimeFunctions.SaveCameraTable(self,True))    
        
        #When the setup items button is clicked in the camera tab  
        self.ui.SetupItems.clicked.connect(lambda : ParallelThreadFunctions.SetupItems(self))       
         
        
        #When the save settings button is clicked
        self.ui.SaveSettings.clicked.connect(lambda : RuntimeFunctions.SaveSettings(self))       
        
        #When the table combobox in the settings tab is selected and its value is changed run this function
        self.ui.SettingNameTableBox.currentIndexChanged.connect(lambda : RuntimeFunctions.SettingNameTableBoxCHANGED(self))    
        
        #When the table combobox in the settings tab is selected and its value is changed run this function
        self.ui.SettingQuantityTableBox.currentIndexChanged.connect(lambda : RuntimeFunctions.SettingQuantityTableBoxCHANGED(self))            
                     
        #When the table combobox in the settings tab is selected and its value is changed run this function
        self.ui.SettingSKUTableBox.currentIndexChanged.connect(lambda : RuntimeFunctions.SettingSKUTableBoxCHANGED(self))     
        
        #When the settings tab is clicked
        self.ui.Settings.clicked.connect(lambda : RuntimeFunctions.SetupSettingsFromFile(self))  
        
        #When the start button is clicked (MAIN LOOP OF PROGRAM)
        self.ui.StartModel.clicked.connect(lambda : ParallelThreadFunctions.StartModel(self)) 
        
        #When the edit item button in the camera tab is clicked
        self.ui.EditItems.clicked.connect(lambda : RuntimeFunctions.EditCameraItems(self)) 
        
        #When a cell in the fullitems table is clicked run this function 
        self.ui.AllModelItemsTable.cellClicked.connect(lambda : RuntimeFunctions.GetCellClickedFullItemTable(self,self.ui.AllModelItemsTable.currentRow()))
                                                       
        #When a cell in the itemsinview table is clicked run this function 
        self.ui.ItemInViewTable.cellClicked.connect(lambda : RuntimeFunctions.GetCellClickedItemsInViewTable(self,self.ui.ItemInViewTable.currentRow()))
        
        #When the add button in the edit items tab is clicked
        self.ui.AddToItemsInView.clicked.connect(lambda : RuntimeFunctions.AddToItemsInView(self))
        
        #When the remove button in the edit items tab is clicked
        self.ui.RemoveFromItemsInView.clicked.connect(lambda : RuntimeFunctions.RemoveFromItemsInView(self))
        
        #When the refresh button in the edit items tab is clicked
        self.ui.RefreshItemsInView.clicked.connect(lambda : RuntimeFunctions.RefreshItemsInView(self))
        
        #When the report tab is clicked
        self.ui.Report.clicked.connect(lambda : RuntimeFunctions.DisplayReport(self))        
        
        #When the remove button in the edit items tab is clicked
        self.ui.SaveItemsInView.clicked.connect(lambda : RuntimeFunctions.SaveItemsInView(self))
        
        #Clear reports
        self.ui.ClearReports.clicked.connect(lambda : ParallelThreadFunctions.ClearReports(self))
        
        #UndoReportChange
        self.ui.UndoReportChange.clicked.connect(lambda : ParallelThreadFunctions.UndoReportChange(self))
        
        #PrintReport
        self.ui.PrintReport.clicked.connect(lambda : ParallelThreadFunctions.PrintReport(self))
        
        #Notification Button
        self.ui.Notification.clicked.connect(lambda : RuntimeFunctions.Notification(self))
            
        #When a cell in the reports table is clicked
        self.ui.ReportTable.cellClicked.connect(lambda : ParallelThreadFunctions.GetCellClickedReportTable(self,self.ui.ReportTable.currentRow()))
        
        #Clear reports
        self.ui.ClearNotifications.clicked.connect(lambda : ParallelThreadFunctions.ClearNotifications(self))
        
        #When the Setup model button is clicked
        self.ui.SetupModel.clicked.connect(lambda : RuntimeFunctions.SetupEditModelComboBoxes(self))
        
        #When the Name combobox in the model setup page is selected and its value is changed run this function
        self.ui.NameComboBox.currentIndexChanged.connect(lambda : RuntimeFunctions.NameComboBoxCHANGED(self))     
        
        #When the ID combobox in the model setup page is selected and its value is changed run this function
        self.ui.IDComboBox.currentIndexChanged.connect(lambda : RuntimeFunctions.IDComboBoxCHANGED(self))   
        
        #When the reset button is clicked in the setup model page
        self.ui.ResetSKUID.clicked.connect(lambda : RuntimeFunctions.ResetSKUID(self))
        
        #When the "Webcam" Radio button is clicked in the Camera tab
        self.ui.WebcamRadio.toggled.connect(lambda: RuntimeFunctions.WebcamRadioToggled(self))
        
        #When the "IPWebcam" Radio button is clicked in the Camera tab
        self.ui.IpWebcamRadio.toggled.connect(lambda: RuntimeFunctions.IpWebcamRadioToggled(self))
        
        #When the help button is clicked
        self.ui.Help.clicked.connect(lambda: webbrowser.open('https://drive.google.com/file/d/1t6BCu1YojVjJiXCIupiUBdNybxlm7R6N/view?usp=sharing'))
        
        #When the About button is clicked
        self.ui.About.clicked.connect(lambda: webbrowser.open('https://drive.google.com/file/d/1PnlIUhGZILnpEbOCB52onB0AB6TheFRA/view?usp=sharing'))
        





########################################################################
## EXECUTE APP
########################################################################
if __name__ == "__main__":
    app = QApplication(sys.argv)
    ########################################################################
    ## 
    ########################################################################
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
########################################################################
## END===>
########################################################################  
