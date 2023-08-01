########################################################################
## IMPORTS
########################################################################
import os, shutil
import mysql.connector
import sys
from PyQt5 import sip
import requests
import cv2
import numpy as np
import imutils
import supervision as sv
import pyodbc as odbc
from pyqt_toast import Toast
from PyQt5.QtGui import QFont
from PyQt5.QtCore import QPoint
########################################################################
# IMPORT GUI FILE
from test import *

#Import Model file
from MODEL import *
########################################################################


#GLOBALS
files = os.listdir('ModelInfo\\imageforkaim') 
database_tables=[]
table_list={}



class DatabaseTables():
    
    def __init__(self, tab, verticalLayout,table):
        self.tab = tab
        self.verticalLayout = verticalLayout
        self.table = table
    
  
class RuntimeFunctions():
    

 
    
    def __init__(self,arg):
        super(RuntimeFunctions,self).__init__()
        self.arg=arg
        
        





                   
            
                            
                            
           
    def SetupModelTable(self):

 
        #Set row count to the amount of classes
        self.ui.modeltable.setRowCount(len(files)) 
           
        row =0
        for classname in files:

            classnamesplitted=classname.split(".")
            classnamesplitted2=classnamesplitted[0].split("-")
                
            prouductname = QtWidgets.QTableWidgetItem()
            prouductname.setText(classnamesplitted2[0] + "-" + classnamesplitted2[1] + "-" + classnamesplitted2[2] + "-" + classnamesplitted2[3])
            
            productid = QtWidgets.QTableWidgetItem()
            productid.setText(classnamesplitted2[4])
            
            prouductname.setFlags(QtCore.Qt.ItemIsEnabled)
            productid.setFlags(QtCore.Qt.ItemIsEnabled)
            
            self.ui.modeltable.setItem(row,0,prouductname)
            self.ui.modeltable.setItem(row,1,productid)
            row+=1
            
    def Setupproductiamge(self,Productsetupcounter):
        Productfullname=files[Productsetupcounter]
        
        productpath="ModelInfo\\imageforkaim\\" + Productfullname
        
        Productinfo=Productfullname.split("-")
        
        objecttype=Productinfo[0] 
        producttype=Productinfo[1]
        companyname=Productinfo[2] 
        productname=Productinfo[3] 
        
        temp=Productinfo[4].split(".")
        productid=temp[0]
        
        #Display text
        self.ui.ObjectType.setText(objecttype)
        self.ui.ProductType.setText(producttype)
        self.ui.CompanyName.setText(companyname)
        self.ui.ProductName.setText(productname)
        self.ui.SkuID.setText(productid)
        
        #Display image
        self.ui.productimage.setPixmap(QtGui.QPixmap(productpath))
        
        #Update image counter
        self.ui.imagecounter.setText(str(Productsetupcounter+1) + "/" + str(len(files)))
        
    def IpWebcamRadioToggled(self):
        self.ui.CameraIpAddress.clear()
        self.ui.CameraIpAddress.setPlaceholderText("192.168.1.4") 
        
    def WebcamRadioToggled(self):
        self.ui.CameraIpAddress.clear()
        self.ui.CameraIpAddress.setPlaceholderText("0")  
                 
    def UpdateCameraTable(self):
        
        self.ui.WebcamRadio.setChecked(True)
        self.ui.CameraIpAddress.clear()
        self.ui.CameraIpAddress.setPlaceholderText("0")
    
         
        
        self.ui.DeleterecordCamera.setVisible(False)
        self.ui.SavechangesCamera.setVisible(False)
        self.ui.RefreshCameraTable.setVisible(False)
        
        self.ui.EditItems.setStyleSheet("background-color: rgb(180, 180, 180); border-radius: 8px;")
        self.ui.SetupItems.setStyleSheet("background-color: rgb(180, 180, 180); border-radius: 8px;")
        self.ui.ViewFootage.setStyleSheet("background-color: rgb(180, 180, 180); border-radius: 8px;")
        
        self.ui.EditItems.setEnabled(False)      
        self.ui.SetupItems.setEnabled(False)
        self.ui.ViewFootage.setEnabled(False)
        
        #Read from file
        with open('CameraInfo\\Cameras.txt', 'r') as f:
            lines = f.readlines()
          
        #Remove all existing record from table
        self.ui.CameraTable.setRowCount(0)   
        #Readd proper row count
        self.ui.CameraTable.setRowCount(len(lines))
        
        
        row=0
        for camerainformation in lines:
            splittedinformation=camerainformation.split(" ") 
            
            Cam_ID = QtWidgets.QTableWidgetItem()
            Cam_ID.setText(splittedinformation[0])
            
            CameraType = QtWidgets.QTableWidgetItem()
            CameraType.setText(splittedinformation[1])
            
            Cam_IP = QtWidgets.QTableWidgetItem()
            Cam_IP.setText(splittedinformation[2])

            
            ItemsInView = QtWidgets.QTableWidgetItem()
            ItemsInView.setText(splittedinformation[3].replace("\n",""))
            
            Cam_IP.setFlags(QtCore.Qt.ItemIsEnabled)
            Cam_ID.setFlags(QtCore.Qt.ItemIsEnabled)
            ItemsInView.setFlags(QtCore.Qt.ItemIsEnabled)
            CameraType.setFlags(QtCore.Qt.ItemIsEnabled)
            
            self.ui.CameraTable.setItem(row,0,Cam_ID)
            self.ui.CameraTable.setItem(row,1,CameraType)
            self.ui.CameraTable.setItem(row,2,Cam_IP)
            self.ui.CameraTable.setItem(row,3,ItemsInView)
            row+=1
        
    def getClickedCell(self, row):
        
        self.ui.DeleterecordCamera.setVisible(True)
        self.ui.SavechangesCamera.setVisible(True)
        self.ui.RefreshCameraTable.setVisible(True)
        
        if(self.ui.CameraTable.item(row, 2).text() != "EMPTY"):
                self.ui.EditItems.setStyleSheet("	background-color:#cc5bce; border-radius: 8px;")
                self.ui.EditItems.setEnabled(True)
        else:
                self.ui.EditItems.setStyleSheet("background-color: rgb(180, 180, 180); border-radius: 8px;")
                self.ui.EditItems.setEnabled(False)
                
        self.ui.SetupItems.setStyleSheet("	background-color:#cc5bce; border-radius: 8px;")
        self.ui.ViewFootage.setStyleSheet("	background-color:#cc5bce; border-radius: 8px;")
                
        self.ui.SetupItems.setEnabled(True)
        self.ui.ViewFootage.setEnabled(True)
        
        print('clicked!', row) 
        print(self.ui.CameraTable.item(row, 0).text())   
        
        
        
        
        
          
        

    def  DeleteCameraRecord(self):
        
        self.ui.CameraTable.removeRow(self.ui.CameraTable.currentRow())
        
    def  SaveCameraTable(self,sendtoast):
        #Clear file
        with open("CameraInfo\\Cameras.txt",'w') as f:
            pass
        
        for row in range(0,self.ui.CameraTable.rowCount()):
            
            Camera_ID=self.ui.CameraTable.item(row, 0).text()
            CameraType=self.ui.CameraTable.item(row, 1).text()
            Camera_IP=self.ui.CameraTable.item(row, 2).text()
            Camera_Items=self.ui.CameraTable.item(row, 3).text()
            
            #Save record to text file
            with open('CameraInfo\\Cameras.txt', 'a') as f:
                f.write(Camera_ID+" "+ CameraType+ " " + Camera_IP +" " + Camera_Items+"\n")
                
        if sendtoast==True:         
            toast=Toast(text='Camera information saved', duration=3, parent=self)
            new_point = QPoint(0, 0)
            toast.setPosition(new_point)
            toast.show()
            
                
            
 
    def  RefreshCameraTable(self):
        
        #Read from file
        with open('CameraInfo\\Cameras.txt', 'r') as f:
            lines = f.readlines()
          
        #Remove all existing record from table
        self.ui.CameraTable.setRowCount(0)   
        #Readd proper row count
        self.ui.CameraTable.setRowCount(len(lines))
        
        
        row=0
        for camerainformation in lines:
            splittedinformation=camerainformation.split(" ") 
            
            Cam_ID = QtWidgets.QTableWidgetItem()
            Cam_ID.setText(splittedinformation[0])

            
            CameraType = QtWidgets.QTableWidgetItem()
            CameraType.setText(splittedinformation[1])
            
            Cam_IP = QtWidgets.QTableWidgetItem()
            Cam_IP.setText(splittedinformation[2])

            
            
            ItemsInView = QtWidgets.QTableWidgetItem()
            ItemsInView.setText(splittedinformation[3].replace("\n",""))
            
            Cam_IP.setFlags(QtCore.Qt.ItemIsEnabled)
            Cam_ID.setFlags(QtCore.Qt.ItemIsEnabled)
            ItemsInView.setFlags(QtCore.Qt.ItemIsEnabled)
            CameraType.setFlags(QtCore.Qt.ItemIsEnabled)
            
            self.ui.CameraTable.setItem(row,0,Cam_ID)
            self.ui.CameraTable.setItem(row,1,CameraType)
            self.ui.CameraTable.setItem(row,2,Cam_IP)
            self.ui.CameraTable.setItem(row,3,ItemsInView)
            row+=1    
                    
        
    def SetupSettingsPage(self):
        global table_list
        table_list={}
        
        #Set up table list dictionary KEY= table name, Value = List of column names
        for i in range(self.ui.tabWidget.count()):
            column_list=[]
            for columnindex in range(0,database_tables[i].table.columnCount()):
                column_list.append(database_tables[i].table.horizontalHeaderItem(columnindex).text())
            # print(database_tables[i].table.horizontalHeaderItem(columnindex).text())
            table_list[self.ui.tabWidget.tabText(i)]=column_list
        
            #Add information to combo list (TABLES)
            self.ui.SettingSKUTableBox.clear()
            self.ui.SettingSKUTableBox.addItems(list(table_list.keys()))
            
            self.ui.SettingQuantityTableBox.clear()
            self.ui.SettingQuantityTableBox.addItems(list(table_list.keys()))
            
            self.ui.SettingNameTableBox.clear()
            self.ui.SettingNameTableBox.addItems(list(table_list.keys()))
            
            #Add information to combo list (COLUMNS)
            
            self.ui.SettingSKUColumnBox.clear()
            self.ui.SettingSKUColumnBox.addItems(table_list[self.ui.SettingSKUTableBox.currentText()])
            
            self.ui.SettingQuantityColumnBox.clear()
            self.ui.SettingQuantityColumnBox.addItems(table_list[self.ui.SettingQuantityTableBox.currentText()])
            
            self.ui.SettingNameColumnBox.clear()
            self.ui.SettingNameColumnBox.addItems(table_list[self.ui.SettingNameTableBox.currentText()])
            
    def SetupEditModelComboBoxes(self):
        global table_list
        table_list={}
        NameList=[]
        IDList=[]
        
        
        #Read from file
        with open('settings\setting.txt', 'r') as f:
            temp = f.readlines()
                            
        settings = [item.strip() for item in temp]



                    
                            
        sku_id_table_index=0
        sku_id_column_index=0
        sku_id_row_index=0
        Name_column_index=0

                            
        for i in range(0,self.ui.tabWidget.count()):
                                
                                if self.ui.tabWidget.tabText(i)==settings[5]:
                                    sku_id_table_index=i
                                    break

                                    

        for columnindex in range(0,database_tables[sku_id_table_index].table.columnCount()):
                                    
                                    if database_tables[sku_id_table_index].table.horizontalHeaderItem(columnindex).text()==settings[8]:
                                        sku_id_column_index=columnindex
                                    if database_tables[sku_id_table_index].table.horizontalHeaderItem(columnindex).text()==settings[10]:
                                        Name_column_index=columnindex
                                        
        for rowindex in range(0,database_tables[sku_id_table_index].table.rowCount()):
                NameList.append(database_tables[sku_id_table_index].table.item(rowindex, Name_column_index).text())
                IDList.append(database_tables[sku_id_table_index].table.item(rowindex, sku_id_column_index).text())
        
        
        
            #Add information to combo list (NAMES)
        self.ui.NameComboBox.clear()
        self.ui.NameComboBox.addItems(NameList)
            
        self.ui.IDComboBox.clear()
        self.ui.IDComboBox.addItems(IDList)

    def NameComboBoxCHANGED(self):   
        self.ui.IDComboBox.setCurrentIndex(self.ui.NameComboBox.currentIndex())
        
    def IDComboBoxCHANGED(self):   
        self.ui.NameComboBox.setCurrentIndex(self.ui.IDComboBox.currentIndex())
        self.ui.SkuID.setText(self.ui.IDComboBox.currentText())
        
    def ResetSKUID(self):
        self.ui.SkuID.setText("000000") 
              
    def SaveSettings(self):
        
        
            radiobuttonvalue=""
            

            if self.ui.radioButton_Realtime.isChecked():
                    radiobuttonvalue=self.ui.radioButton_Realtime.text()


            elif self.ui.radioButton_10s.isChecked():
                        radiobuttonvalue=self.ui.radioButton_10s.text()

                    
            elif self.ui.radioButton_20s.isChecked():
                        radiobuttonvalue=self.ui.radioButton_20s.text()
                
                    
            elif self.ui.radioButton_30s.isChecked():
                        radiobuttonvalue=self.ui.radioButton_30s.text()

                    
            elif self.ui.radioButton_1m.isChecked():
                        radiobuttonvalue=self.ui.radioButton_1m.text()
                
                    
            elif self.ui.radioButton_2m.isChecked():
                        radiobuttonvalue=self.ui.radioButton_2m.text()
            
                    
            elif self.ui.radioButton_5m.isChecked():
                        radiobuttonvalue=self.ui.radioButton_5m.text()

            elif self.ui.radioButton_10m.isChecked():
                        radiobuttonvalue=self.ui.radioButton_10m.text()

            elif self.ui.radioButton_30m.isChecked():
                        radiobuttonvalue=self.ui.radioButton_30m.text()
                        
            #Delete old settings if they exist and add the new ones         
            with open('settings\setting.txt', 'r') as f:           
                lines = f.readlines()
                
            with open('settings\setting.txt', 'w') as f:
                f.write(lines[0] +lines[1] + lines[2] + lines[3] + lines[4]+self.ui.SettingSKUTableBox.currentText()+"\n"+ self.ui.SettingQuantityTableBox.currentText() +"\n" + self.ui.SettingNameTableBox.currentText()+"\n" + self.ui.SettingSKUColumnBox.currentText()+
                        "\n" + self.ui.SettingQuantityColumnBox.currentText()+"\n" + self.ui.SettingNameColumnBox.currentText()+"\n" + radiobuttonvalue)

            toast=Toast(text='Settings Saved', duration=3, parent=self)
            new_point = QPoint(0, 0)
            toast.setPosition(new_point)
            toast.show()
            RuntimeFunctions.SaveCurrentQuantityValues(self)
        
    def SaveCurrentQuantityValues(self):
        
        
            #Read from file
            with open('settings\setting.txt', 'r') as f:
                temp = f.readlines()
                                
            settings = [item.strip() for item in temp]
                                    
            #If db not setup then return
            try:
                        
                        server = settings[0]
                        port=settings[1]
                        database = settings[2]
                        username = settings[3]
                        password = settings[4]


                
                        #Connect to database
                        conn = mysql.connector.connect(user=username, password=password, host=server, database=database,port=int(port))
                        
            except: 
                    return
                    #If settings is not setup return
            try:
                        temp1=settings[5]
                        temp2=settings[6]
                        temp3=settings[7]
                        temp4=settings[8]
                        temp5=settings[9]
                        temp6=settings[10]
                        temp7=settings[11]
                        
            except:
                    return
            
            sku_id_table_index=0
            sku_id_column_index=0
            sku_id_row_index=0
            quantity_column_index=0
            quantity_table_index=0
                            
            for i in range(0,self.ui.tabWidget.count()):
                                
                if self.ui.tabWidget.tabText(i)==settings[5]:
                    sku_id_table_index=i
                if self.ui.tabWidget.tabText(i)==settings[6]:
                    quantity_table_index=i
                                    

            for columnindex in range(0,database_tables[quantity_table_index].table.columnCount()):
                                    
                    if database_tables[quantity_table_index].table.horizontalHeaderItem(columnindex).text()==settings[8]:
                        sku_id_column_index=columnindex
                    if database_tables[quantity_table_index].table.horizontalHeaderItem(columnindex).text()==settings[9]:
                        quantity_column_index=columnindex            
            
            
            
            
            DB_Quanity= open('settings\\DB_Quantity.txt','w')    
               
            for rowindex in range(0,database_tables[quantity_table_index].table.rowCount()):
                DB_Quanity.write(database_tables[quantity_table_index].table.item(rowindex, quantity_column_index).text()+"\n")
                
            DB_Quanity.close()
                
        
               
    def SettingNameTableBoxCHANGED(self):
        
        try:
            self.ui.SettingNameColumnBox.clear()
            self.ui.SettingNameColumnBox.addItems(table_list[self.ui.SettingNameTableBox.currentText()])
        except:
            print("")
        
    def SettingQuantityTableBoxCHANGED(self):
        

        try:       
            self.ui.SettingQuantityColumnBox.clear()
            self.ui.SettingQuantityColumnBox.addItems(table_list[self.ui.SettingQuantityTableBox.currentText()])
        except:
            print("")
        
    def SettingSKUTableBoxCHANGED(self):

        try:     
            self.ui.SettingSKUColumnBox.clear()
            self.ui.SettingSKUColumnBox.addItems(table_list[self.ui.SettingSKUTableBox.currentText()])    
        except:
            print("")
            
            
    def SetupSettingsFromFile(self):
        
        #Read from file
        with open('settings\setting.txt', 'r') as f:
            temp = f.readlines()
            
            settings = [item.strip() for item in temp]

                
                
            try:
                temp=settings[5]

                self.ui.SettingSKUTableBox.setCurrentText(settings[5])
                self.ui.SettingQuantityTableBox.setCurrentText(settings[6])
                self.ui.SettingNameTableBox.setCurrentText(settings[7])
                
                self.ui.SettingSKUColumnBox.setCurrentText(settings[8])
                self.ui.SettingQuantityColumnBox.setCurrentText(settings[9])
                self.ui.SettingNameColumnBox.setCurrentText(settings[10])
                
                #Uncheck all radio buttons
                
                self.ui.radioButton_Realtime.setChecked(False)
                self.ui.radioButton_10s.setChecked(False)
                self.ui.radioButton_20s.setChecked(False)
                self.ui.radioButton_30s.setChecked(False)
                self.ui.radioButton_1m.setChecked(False)
                self.ui.radioButton_2m.setChecked(False)
                self.ui.radioButton_5m.setChecked(False)
                self.ui.radioButton_10m.setChecked(False)
                self.ui.radioButton_30m.setChecked(False)

                #Set the one saved in the settings file to checked
                exec("self.ui.radioButton_%s.setChecked(True)"%(settings[11]))
                
            except:
                print("No pre-existing settings to load")



    def EditCameraItems(self):
        
        self.ui.AddToItemsInView.setStyleSheet("background-color: rgb(180, 180, 180); border-radius: 8px;")
        self.ui.AddToItemsInView.setEnabled(False)    
        
        self.ui.RemoveFromItemsInView.setStyleSheet("background-color: rgb(180, 180, 180); border-radius: 8px;")
        self.ui.RemoveFromItemsInView.setEnabled(False)    
        
        self.ui.SaveItemsInView.setStyleSheet("background-color: rgb(180, 180, 180); border-radius: 8px;")
        self.ui.SaveItemsInView.setEnabled(False)    
        
        self.ui.RefreshItemsInView.setStyleSheet("background-color: rgb(180, 180, 180); border-radius: 8px;")
        self.ui.RefreshItemsInView.setEnabled(False)  

        #Setup full items table (right one)
        
        #Set row count to the amount of classes
        self.ui.AllModelItemsTable.setRowCount(len(files)) 
           
        row =0
        for classname in files:

            classnamesplitted=classname.split(".")
            classnamesplitted2=classnamesplitted[0].split("-")
                
            prouductname = QtWidgets.QTableWidgetItem()
            prouductname.setText(classnamesplitted2[0] + "-" + classnamesplitted2[1] + "-" + classnamesplitted2[2] + "-" + classnamesplitted2[3])
            
            
            prouductname.setFlags(QtCore.Qt.ItemIsEnabled)

            
            self.ui.AllModelItemsTable.setItem(row,0,prouductname)
            row+=1 
            
            
            
            
        #Setup Items in view table (Left One)
        
        ItemsinView=self.ui.CameraTable.item(self.ui.CameraTable.currentRow(), 3).text()
        ItemsinViewList=ItemsinView.split(",")
        
        self.ui.ItemInViewTable.setRowCount(len(ItemsinViewList)) 
        
        
        row=0
        for item in ItemsinViewList:
            
            ItemsinViewToAdd = QtWidgets.QTableWidgetItem()
            ItemsinViewToAdd.setText(item)
            ItemsinViewToAdd.setFlags(QtCore.Qt.ItemIsEnabled)
            
            self.ui.ItemInViewTable.setItem(row,0,ItemsinViewToAdd)
            
            row+=1
            
    def GetCellClickedFullItemTable(self,row):
        self.ui.AddToItemsInView.setStyleSheet("background-color:#cc5bce; border-radius: 8px;")
        self.ui.AddToItemsInView.setEnabled(True)    
        
        
    def GetCellClickedItemsInViewTable(self,row):
        self.ui.RemoveFromItemsInView.setStyleSheet("background-color:#cc5bce; border-radius: 8px;")
        self.ui.RemoveFromItemsInView.setEnabled(True)  
        
        self.ui.SaveItemsInView.setStyleSheet("background-color:#cc5bce; border-radius: 8px;")
        self.ui.SaveItemsInView.setEnabled(True) 
        
        self.ui.RefreshItemsInView.setStyleSheet("background-color:#cc5bce; border-radius: 8px;")
        self.ui.RefreshItemsInView.setEnabled(True)
        
         
    def AddToItemsInView(self):
        
        ItemToAdd = QtWidgets.QTableWidgetItem()
        ItemToAdd.setText(self.ui.AllModelItemsTable.item(self.ui.AllModelItemsTable.currentRow(), 0).text())
        ItemToAdd.setFlags(QtCore.Qt.ItemIsEnabled)
        
        self.ui.ItemInViewTable.setRowCount(self.ui.ItemInViewTable.rowCount()+1)                            
                        

        self.ui.ItemInViewTable.setItem(self.ui.ItemInViewTable.rowCount()-1,0,ItemToAdd)

                
        #Save record to text file
        #with open('CameraInfo\\Cameras.txt', 'a') as f:
            #f.write(Cam_ID.text()+" "+ Cam_IP.text() +" " + ItemsInView.text()+"\n") 
    def RemoveFromItemsInView(self):
        self.ui.ItemInViewTable.removeRow(self.ui.ItemInViewTable.currentRow())
        
        
    def RefreshItemsInView(self):
        
        #Remove all existing record from table
        self.ui.ItemInViewTable.setRowCount(0)   

        
        ItemsinView=self.ui.CameraTable.item(self.ui.CameraTable.currentRow(), 3).text()
        ItemsinViewList=ItemsinView.split(",")
        
        self.ui.ItemInViewTable.setRowCount(len(ItemsinViewList)) 
        
        
        row=0
        for item in ItemsinViewList:
            
            ItemsinViewToAdd = QtWidgets.QTableWidgetItem()
            ItemsinViewToAdd.setText(item)
            ItemsinViewToAdd.setFlags(QtCore.Qt.ItemIsEnabled)
            
            self.ui.ItemInViewTable.setItem(row,0,ItemsinViewToAdd)
            
            row+=1
        
        
    def SaveItemsInView(self):
        #Get items from table
        ItemListString=""
        #Clear file
        with open('CameraInfo\\Cameras.txt', 'r') as f:
            lines = f.readlines()
            
        
            
            
        with open("CameraInfo\\Cameras.txt",'w') as f:
            pass
        
        itemlistisempty=False
        
        try:
            ItemListString= ItemListString + self.ui.ItemInViewTable.item(0, 0).text()
        except:
            ItemListString="EMPTY"
            itemlistisempty=True
            
        if itemlistisempty==False:
            for row in range(1,self.ui.ItemInViewTable.rowCount()):    
                ItemListString= ItemListString + "," + self.ui.ItemInViewTable.item(row, 0).text()

            
        #Save record to text file
        with open('CameraInfo\\Cameras.txt', 'a') as f:
            counter=0
            for line in lines:
                if(counter==self.ui.CameraTable.currentRow()):
                    linesplitted=line.split(" ")
                    f.write(linesplitted[0]+" "+ linesplitted[1] +" " +linesplitted[2]+ " " +ItemListString+"\n")
                else:
                    f.write(line)
                counter+=1
                
        #Refresh camera tab information.     
        RuntimeFunctions.UpdateCameraTable(self)
        
    def DisplayReport(self):
        
        self.ui.UndoReportChange.setStyleSheet("background-color: rgb(180, 180, 180); border-radius: 8px;")
        self.ui.UndoReportChange.setEnabled(False)   
    
    def Notification(self):
        self.ui.NotificationNumber.setText("0")
        self.ui.NotificationFrame.setStyleSheet("background-color: transparent; border-radius: 12px;")
        
        if self.ui.NotificationNumber.text()=="0":
            self.ui.NotificationNumber.setVisible(False)
                
