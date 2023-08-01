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
import threading
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
from datetime import datetime, timedelta
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate,Table,TableStyle
import math
import random
import string
########################################################################
# IMPORT GUI FILE
from test import *
#Import Model file
from MODEL import *

########################################################################

#Import loading gif class
from waitingspinnerwidget import QtWaitingSpinner

########################################################################
# IMPORT Custom widgets
from Custom_Widgets.Widgets import *
# INITIALIZE APP SETTINGS
settings = QSettings()
from RuntimeFunctions import *


#Globals
runthread=False
mutex = QMutex()


#Signal class for communicating with the main thread
class Signals(QObject):
    ErrorSignal = Signal(str,str)
    Finished=Signal(int)
    ChangeNotification=Signal(int)
    
    
   


   
    
    
    
    
class bkgTask(QRunnable):
    

    
        
    def __init__(self, SKUID_TABLE,SKUID_COLUMN,QUANTITY_TABLE,QUANTITY_COLUMN,NAME_TABLE,NAME_COLUMN,FREQUENCY,CameraIP,CameraID,mainwindowcontext,CameraType,itemsinviewarray):
       super().__init__()
       
       self.mainwindow=mainwindowcontext
       
       self.ItemsInView=itemsinviewarray
         
       self.signals = Signals()
       self.SKUID_TABLE=SKUID_TABLE
       self.SKUID_COLUMN=SKUID_COLUMN
       
       self.QUANTITY_TABLE=QUANTITY_TABLE
       self.QUANTITY_COLUMN=QUANTITY_COLUMN
        
       self.NAME_TABLE=NAME_TABLE
       self.NAME_COLUMN=NAME_COLUMN
       
       self.FREQUENCY=FREQUENCY
       self.CameraIP=CameraIP
       
       self.CameraID=CameraID
       
       self.CameraType=CameraType
       
       self.OLD_IMAGE=None
       self.OLD_IMAGE_COORDINATES=None
       
       self.skipthisiteration=True
       self.toptext=""
       self.informativetext=""
       
       self.exit=False
    def run(self):
        global runthread
        print("STARTED! Camera "+self.CameraID)
        #initialize YOLO model: 
        YoloModelPath="Models\\YoloModel.pt"
        YOLO_model = YOLO(YoloModelPath)
        Yolo_model_person_detector = YOLO("Models\\yolov8m.pt")
        
        #Initiliaze classnames for resnet model
        with open('Models\\Classnames.txt', 'r') as f:
            Classnames = [line.strip() for line in f]
            
            if self.CameraType=="Webcam": 
                #Initalize camera
                cap = cv2.VideoCapture(int(self.CameraIP))
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 0)
                cap.set(3, 1280)
                cap.set(4, 720)
            
            
            
        
        while(runthread):
            if  self.skipthisiteration==False:
                #Sleep for the time set in the settings tab
                if self.FREQUENCY=='Realtime':
                    time.sleep(0)
                    if(runthread==False):
                        self.exit=True 
                        
                elif self.FREQUENCY=='10s':
                    
                    now = datetime.today()
                    result = now + timedelta(seconds=10)

                    while(True):

                        now = datetime.today()    
                        if now>result:
                            break
                        
                        time.sleep(2)
                        
                        if(runthread==False):
                            self.exit=True
                        

                elif self.FREQUENCY=='20s':
                    
                    now = datetime.today()
                    result = now + timedelta(seconds=20)

                    while(True):

                        now = datetime.today()    
                        if now>result:
                            break
                        
                        time.sleep(2)
                        
                        if(runthread==False):
                            self.exit=True
                            break
                                         
                elif self.FREQUENCY=='30s':
                    
                    now = datetime.today()

                    result = now + timedelta(seconds=30)

                    while(True):     
                        now = datetime.today()         
                        if now>result:
                            break
                        
                        time.sleep(2)
                        
                        if(runthread==False):
                            self.exit=True  
                            break  
                                     
                elif self.FREQUENCY=='1m':
                    
                    now = datetime.today()
                    result = now + timedelta(minutes=1)

                    while(True):

                        now = datetime.today()    
                        if now>result:
                            break
                        
                        time.sleep(2)
                        
                        if(runthread==False):
                            self.exit=True   
                            break 
                                         
                elif self.FREQUENCY=='2m':
                    
                    now = datetime.today()
                    result = now + timedelta(minutes=2)

                    while(True):

                        now = datetime.today()    
                        if now>result:
                            break
                        
                        time.sleep(2)
                        
                        if(runthread==False):
                            self.exit=True    
                            break
                                  
                elif self.FREQUENCY=='5m':
                    
                    now = datetime.today()
                    result = now + timedelta(minutes=3)

                    while(True):

                        now = datetime.today()    
                        if now>result:
                            break
                        
                        time.sleep(2)
                        
                        if(runthread==False):
                            self.exit=True 
                            break  
                                            
                elif self.FREQUENCY=='10m':
                    
                    now = datetime.today()
                    result = now + timedelta(minutes=10)

                    while(True):

                        now = datetime.today()    
                        if now>result:
                            break
                        
                        time.sleep(2)
                        
                        if(runthread==False):
                            self.exit=True 
                            break  
                                    
                elif self.FREQUENCY=='30m':
                    
                    now = datetime.today()
                    result = now + timedelta(minutes=30)

                    while(True):

                        now = datetime.today()    
                        if now>result:
                            break
                        
                        time.sleep(2)
                        
                        if(runthread==False):
                            self.exit=True 
                            break              
                
                      
            if(self.exit==True or runthread==False):

                    print("Camera" + str(self.CameraID) +" Successfully closed") 

                    break     
                
                
                
                #Get snapshot from camera
            try:
                    if self.CameraType=="IpWebcam":
                        #Try connecting to the camera if it fails through and error and close the thread
                        img_resp = requests.get("http://"+self.CameraIP+":8080//shot.jpg",timeout=10)
                        img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
                        img = cv2.imdecode(img_arr, -1)
                        img = imutils.resize(img, width=1280, height=720)
                        print("Snapshot taken: Camera ID= " + str(self.CameraID))  
                    else:
                        frameisread,img = cap.read()
                        
                        now = datetime.today()
                        result = now + timedelta(seconds=5)
                        
                        #Try to reread image for 5 secs if it fails the first time.
                        while frameisread==False:
                            now = datetime.today()
                            if now > result:
                                break
                            frameisread,img = cap.read()
                        
                        
                        img = imutils.resize(img, width=1280, height=720)
                        print("Snapshot taken: Camera ID= " + str(self.CameraID)) 

                    
            except:
                mutex.lock()
                self.toptext="Camera Error"
                self.informativetext='Failed to connect to camera with IP/Index =' + self.CameraIP+"\nCamera will no longer recieve input.\nOther Cameras will remain unaffected."
                mutex.unlock()
                self.signals.ErrorSignal.emit(self.toptext,self.informativetext)

                break
                


            
            ##YOLO

            mutex.lock()
            detection_output_Person = Yolo_model_person_detector.predict(source=img, conf=0.6, save=False,device='cpu')
            mutex.unlock()
            
            #Check if theres a person in the frame. If there is, skip this frame
            result_person = detection_output_Person[0]
            
            PersonDetected=False
            for box in result_person.boxes:
                class_id = box.cls[0].item()
                if class_id == 0:
                    print("Person detected in Camera "+str(self.CameraID)+ "...Skipping processing for this image")
                    PersonDetected=True
                    break
                
            if(PersonDetected==True):
                continue
                

            if self.skipthisiteration==False:

                current_output=detection_output_Person[0]
                old_image_output=self.OLD_IMAGE_COORDINATES[0]

                ObjectsFoundFromNewImage=[]
                #Iterate through the results
                for old_coordinate in old_image_output.boxes:
                    class_id = old_coordinate.cls[0].item()
                    if class_id == 39.0:
                        
                        if runthread==False:
                            self.exit=True 
                            break      
                        
                        Matchfound=False

                        
                        old_cords = old_coordinate.xyxy[0].tolist()
                        x1_old = old_cords[0]-50
                        y1_old = old_cords[1]-100
                        x2_old = old_cords[2]+50
                        y2_old = old_cords[3]+50
                        
                        
                        counter=0 
                        for new_coordinate in current_output.boxes:
                            class_id2 = new_coordinate.cls[0].item()
                            if class_id2 == 39.0:
                                
                                new_cords = new_coordinate.xyxy[0].tolist()                     
                                x1_new = new_cords[0]
                                y1_new = new_cords[1]
                                x2_new = new_cords[2]
                                y2_new = new_cords[3]
                                
                                if  (x1_new>=x1_old and x1_new<=x2_old) and (y1_new>=y1_old and y1_new<=y2_old) and (x2_new<=x2_old and x2_new>=x1_old) and (y2_new<=y2_old and y2_new>=y1_old):  
                                    ObjectsFoundFromNewImage.append(counter) 
                                    Matchfound=True                    
                                    break
                                counter+=1

                            

                        
                            
                        #That means that the product is no longer in view
                        #Take it coordiantes and crop the image out then pass it to the classification model    
                        if Matchfound==False:   
                            print("Detected change updating database...")
                                            # the computation device
                            device = 'cpu'
                            
                            # initialize the model and load the trained weights
                            RESNET_model = MODEL.build_model(
                                pretrained=False, fine_tune=False, num_classes=len(Classnames)
                            ).to(device)
                            print('[INFO]: Loading custom-trained weights...')
                            checkpoint = torch.load('Models\\Resnet34Model2.pth', map_location=device)
                            RESNET_model.load_state_dict(checkpoint['model_state_dict'])
                            RESNET_model.eval()
                            
                            # define preprocess transforms
                            transform = transforms.Compose([
                                transforms.ToPILImage(),
                                transforms.Resize((224, 224)),
                                transforms.ToTensor(),
                                transforms.Normalize(
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]
                                )
                            ]) 
                            
                            #Convert numpy array to PIL image
                            
                            #Crop image
                            img_cropped = self.OLD_IMAGE[int(y1_old+100):int(y2_old-50), int(x1_old+50):int(x2_old-50)]
                            
                            
                            # convert to RGB format
                            img_cropped = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2RGB)
                            img_cropped = transform(img_cropped)
                            # add batch dimension
                            img_cropped = torch.unsqueeze(img_cropped, 0)
                            with torch.no_grad():
                                outputs = RESNET_model(img_cropped.to(device))
                            output_label = torch.topk(outputs, 1)
                            pred_class = Classnames[int(output_label.indices)]  
                            
                            #Output
                            print("Prediction: " + pred_class)
                            
                            SKU_ID=""
                            
                            #Read image names to find SKU ID
                            for imagename in files:
                                imagename_splitted=imagename.split(".")
                                imagename_splitted2=imagename_splitted[0].split("-")
                                productname=imagename_splitted2[0]+"-"+imagename_splitted2[1]+"-"+imagename_splitted2[2]+"-"+imagename_splitted2[3]
                                
                                if productname==pred_class:
                                    SKU_ID=imagename_splitted2[4]
                                    break
                                
                                #If the ID is it's default value i.e never changed then discard this image since theres no way to update the database.
                            if(SKU_ID=="000000"):
                                
                                
                                NotificationType = QtWidgets.QTableWidgetItem()
                                NotificationType.setText("Missing SKU ID ")

                                
                                TimeStamp = QtWidgets.QTableWidgetItem()
                                TimeStamp.setText(str(datetime.today()))
                                
                                Alert = QtWidgets.QTableWidgetItem()
                                Alert.setText("Product: "+pred_class+" Change detected but SKUID was not changed from the default value of 0000000 thus no update took place")
                                
                                NotificationType.setFlags(QtCore.Qt.ItemIsEnabled)
                                TimeStamp.setFlags(QtCore.Qt.ItemIsEnabled)
                                Alert.setFlags(QtCore.Qt.ItemIsEnabled)
                                
                                mutex.lock()
                                
                                self.mainwindow.ui.NotificationTable.setRowCount(self.mainwindow.ui.NotificationTable.rowCount()+1)   
                                
                                self.mainwindow.ui.NotificationTable.setItem(self.mainwindow.ui.NotificationTable.rowCount()-1,0,NotificationType)
                                self.mainwindow.ui.NotificationTable.setItem(self.mainwindow.ui.NotificationTable.rowCount()-1,1,TimeStamp)
                                self.mainwindow.ui.NotificationTable.setItem(self.mainwindow.ui.NotificationTable.rowCount()-1,2,Alert)
                                
                                #Increment Notification number and change BG color
                                self.signals.ChangeNotification.emit(1)
                                mutex.unlock()
                                continue
                            else:
                                #SKU ID Is different so update database
                                #Get column and row indices to know which value to update in the database table
                                
                                #Read from file
                                with open('settings\setting.txt', 'r') as f:
                                    temp = f.readlines()
                                
                                    settings = [item.strip() for item in temp]
                                    
                                server = settings[0]
                                port=settings[1]
                                database = settings[2]
                                username = settings[3]
                                password = settings[4]


                        
                                
                                sku_id_table_index=0
                                sku_id_column_index=0
                                sku_id_row_index=0
                                quantity_column_index=0
                                quantity_table_index=0
                                
                                for i in range(0,self.mainwindow.ui.tabWidget.count()):
                                    
                                    if self.mainwindow.ui.tabWidget.tabText(i)==self.SKUID_TABLE:
                                        sku_id_table_index=i
                                    if self.mainwindow.ui.tabWidget.tabText(i)==self.QUANTITY_TABLE:
                                        quantity_table_index=i
                                        

                                for columnindex in range(0,database_tables[sku_id_table_index].table.columnCount()):
                                        
                                        if database_tables[sku_id_table_index].table.horizontalHeaderItem(columnindex).text()==self.SKUID_COLUMN:
                                            sku_id_column_index=columnindex
                                        if database_tables[sku_id_table_index].table.horizontalHeaderItem(columnindex).text()==self.QUANTITY_COLUMN:
                                            quantity_column_index=columnindex
                                            
                                for rowindex in range(0,database_tables[sku_id_table_index].table.rowCount()):

                                        if database_tables[sku_id_table_index].table.item(rowindex, sku_id_column_index).text()==SKU_ID:
                                            sku_id_row_index= rowindex    
                                            break
                                        
                                #Update table values
                                new_item = QtWidgets.QTableWidgetItem()
                                new_item.setText(str(int(database_tables[sku_id_table_index].table.item(sku_id_row_index, quantity_column_index).text())-1))  
                                new_item.setFlags(QtCore.Qt.ItemIsEnabled)  
                                
                                #Add report to Reports table
                                self.mainwindow.ui.ReportTable.setRowCount(self.mainwindow.ui.ReportTable.rowCount()+1)                            
                            
                                ReportType = QtWidgets.QTableWidgetItem()
                                ReportType.setText("Product Subtraction")

                                
                                TimeStamp = QtWidgets.QTableWidgetItem()
                                TimeStamp.setText(str(datetime.today()))
                                
                                Change = QtWidgets.QTableWidgetItem()
                                Change.setText("Classname: " + pred_class + " ID: " + SKU_ID)
                                
                                ReportType.setFlags(QtCore.Qt.ItemIsEnabled)
                                TimeStamp.setFlags(QtCore.Qt.ItemIsEnabled)
                                Change.setFlags(QtCore.Qt.ItemIsEnabled)
                                
                                mutex.lock()
                                #Add to report table
                                self.mainwindow.ui.ReportTable.setItem(self.mainwindow.ui.ReportTable.rowCount()-1,0,ReportType)
                                self.mainwindow.ui.ReportTable.setItem(self.mainwindow.ui.ReportTable.rowCount()-1,1,TimeStamp)
                                self.mainwindow.ui.ReportTable.setItem(self.mainwindow.ui.ReportTable.rowCount()-1,2,Change)
                                
                                #Update database table
                                database_tables[sku_id_table_index].table.setItem(sku_id_row_index, quantity_column_index,new_item)  
                                mutex.unlock() 
                            

                  
                  
                  
                  
                  
                    
                if(self.exit==True or runthread==False):

                    print("Camera" + str(self.CameraID) +" Successfully closed") 

                    break                    
                  
                  
                 
                  
                #Finished Iteration. Now check if there are any new bounding boxes in the new imagem if yes then ADD that classify that object and add it to the database    
                counter=0 
                for new_coordinate in current_output.boxes:
                    class_id = new_coordinate.cls[0].item()
                    if class_id == 39.0:    
                            if(runthread==False):
                                self.exit=True 
                                break    
                        
                            if counter in ObjectsFoundFromNewImage:
                                counter+=1 
                                continue
                            
                            #If code reaches this point this means that theres an object in the new image thats not in the old image.
                            new_cords = new_coordinate.xyxy[0].tolist()                     
                            x1_new = new_cords[0]
                            y1_new = new_cords[1]
                            x2_new = new_cords[2]
                            y2_new = new_cords[3]
                            
                            
                            
                            
                            
                            
                            
                            
                            print("Detected change updating database...")
                                            # the computation device
                            device = 'cpu'
                            
                            # initialize the model and load the trained weights
                            RESNET_model = MODEL.build_model(
                                pretrained=False, fine_tune=False, num_classes=len(Classnames)
                            ).to(device)
                            print('[INFO]: Loading custom-trained weights...')
                            checkpoint = torch.load('Models\\Resnet34Model2.pth', map_location=device)
                            RESNET_model.load_state_dict(checkpoint['model_state_dict'])
                            RESNET_model.eval()
                            
                            # define preprocess transforms
                            transform = transforms.Compose([
                                transforms.ToPILImage(),
                                transforms.Resize((224, 224)),
                                transforms.ToTensor(),
                                transforms.Normalize(
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]
                                )
                            ]) 
                            
                            #Convert numpy array to PIL image
                            
                            #Crop image
                            img_cropped = img[int(y1_new):int(y2_new), int(x1_new):int(x2_new)]
                            
                            
                            # convert to RGB format
                            img_cropped = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2RGB)
                            img_cropped = transform(img_cropped)
                            # add batch dimension
                            img_cropped = torch.unsqueeze(img_cropped, 0)
                            with torch.no_grad():
                                outputs = RESNET_model(img_cropped.to(device))
                            output_label = torch.topk(outputs, 1)
                            pred_class = Classnames[int(output_label.indices)]  
                            
                            #Output
                            print("Prediction: " + pred_class)
                            
                            SKU_ID=""
                            
                            #Read image names to find SKU ID
                            for imagename in files:
                                imagename_splitted=imagename.split(".")
                                imagename_splitted2=imagename_splitted[0].split("-")
                                productname=imagename_splitted2[0]+"-"+imagename_splitted2[1]+"-"+imagename_splitted2[2]+"-"+imagename_splitted2[3]
                                
                                if productname==pred_class:
                                    SKU_ID=imagename_splitted2[4]
                                    break
                            
                            
                            
                            #Check if the class is supposed to be in view of the camera if not then send a notifcation regarding that
                            if pred_class not in self.ItemsInView:
                                

                                NotificationType = QtWidgets.QTableWidgetItem()
                                NotificationType.setText("Misplaced Item")

                                
                                TimeStamp = QtWidgets.QTableWidgetItem()
                                TimeStamp.setText(str(datetime.today()))
                                
                                Alert = QtWidgets.QTableWidgetItem()
                                Alert.setText("Camera: "+self.CameraID+" Detected a misplaced product. Product name: "+ pred_class)
                                
                                NotificationType.setFlags(QtCore.Qt.ItemIsEnabled)
                                TimeStamp.setFlags(QtCore.Qt.ItemIsEnabled)
                                Alert.setFlags(QtCore.Qt.ItemIsEnabled)
                                
                                mutex.lock()
                                
                                self.mainwindow.ui.NotificationTable.setRowCount(self.mainwindow.ui.NotificationTable.rowCount()+1)   
                                
                                self.mainwindow.ui.NotificationTable.setItem(self.mainwindow.ui.NotificationTable.rowCount()-1,0,NotificationType)
                                self.mainwindow.ui.NotificationTable.setItem(self.mainwindow.ui.NotificationTable.rowCount()-1,1,TimeStamp)
                                self.mainwindow.ui.NotificationTable.setItem(self.mainwindow.ui.NotificationTable.rowCount()-1,2,Alert)
                                
                                #Increment Notification number and change BG color
                                self.signals.ChangeNotification.emit(1)
                                mutex.unlock()                         
                            
                            
                            
                            
                                
                            #If the ID is it's default value i.e never changed then discard this image since theres no way to update the database.
                            if(SKU_ID=="000000"):
                                
                                
                                NotificationType = QtWidgets.QTableWidgetItem()
                                NotificationType.setText("Missing SKU ID ")

                                
                                TimeStamp = QtWidgets.QTableWidgetItem()
                                TimeStamp.setText(str(datetime.today()))
                                
                                Alert = QtWidgets.QTableWidgetItem()
                                Alert.setText("Product: "+pred_class+" Change detected but SKUID was not changed from the default value of 0000000 thus no update took place")
                                
                                NotificationType.setFlags(QtCore.Qt.ItemIsEnabled)
                                TimeStamp.setFlags(QtCore.Qt.ItemIsEnabled)
                                Alert.setFlags(QtCore.Qt.ItemIsEnabled)
                                
                                mutex.lock()
                                
                                self.mainwindow.ui.NotificationTable.setRowCount(self.mainwindow.ui.NotificationTable.rowCount()+1)   
                                
                                self.mainwindow.ui.NotificationTable.setItem(self.mainwindow.ui.NotificationTable.rowCount()-1,0,NotificationType)
                                self.mainwindow.ui.NotificationTable.setItem(self.mainwindow.ui.NotificationTable.rowCount()-1,1,TimeStamp)
                                self.mainwindow.ui.NotificationTable.setItem(self.mainwindow.ui.NotificationTable.rowCount()-1,2,Alert)
                                
                                #Increment Notification number and change BG color
                                self.signals.ChangeNotification.emit(1)
                                mutex.unlock()
                                counter+=1
                                continue
                            else:
                                #SKU ID Is different so update database
                                #Get column and row indices to know which value to update in the database table
                                
                                #Read from file
                                with open('settings\setting.txt', 'r') as f:
                                    temp = f.readlines()
                                
                                    settings = [item.strip() for item in temp]
                                    
                                server = settings[0]
                                port=settings[1]
                                database = settings[2]
                                username = settings[3]
                                password = settings[4]


                        
                                
                                sku_id_table_index=0
                                sku_id_column_index=0
                                sku_id_row_index=0
                                quantity_column_index=0
                                quantity_table_index=0
                                
                                for i in range(0,self.mainwindow.ui.tabWidget.count()):
                                    
                                    if self.mainwindow.ui.tabWidget.tabText(i)==self.SKUID_TABLE:
                                        sku_id_table_index=i
                                    if self.mainwindow.ui.tabWidget.tabText(i)==self.QUANTITY_TABLE:
                                        quantity_table_index=i
                                        

                                for columnindex in range(0,database_tables[sku_id_table_index].table.columnCount()):
                                        
                                        if database_tables[sku_id_table_index].table.horizontalHeaderItem(columnindex).text()==self.SKUID_COLUMN:
                                            sku_id_column_index=columnindex
                                        if database_tables[sku_id_table_index].table.horizontalHeaderItem(columnindex).text()==self.QUANTITY_COLUMN:
                                            quantity_column_index=columnindex
                                            
                                for rowindex in range(0,database_tables[sku_id_table_index].table.rowCount()):

                                        if database_tables[sku_id_table_index].table.item(rowindex, sku_id_column_index).text()==SKU_ID:
                                            sku_id_row_index= rowindex    
                                            break
                                        
                                #Update table values
                                new_item = QtWidgets.QTableWidgetItem()
                                new_item.setText(str(int(database_tables[sku_id_table_index].table.item(sku_id_row_index, quantity_column_index).text())+1))  
                                new_item.setFlags(QtCore.Qt.ItemIsEnabled)  
                                
                                #Add report to Reports table
                                self.mainwindow.ui.ReportTable.setRowCount(self.mainwindow.ui.ReportTable.rowCount()+1)                            
                            
                                ReportType = QtWidgets.QTableWidgetItem()
                                ReportType.setText("Product Addition")

                                
                                TimeStamp = QtWidgets.QTableWidgetItem()
                                TimeStamp.setText(str(datetime.today()))
                                
                                Change = QtWidgets.QTableWidgetItem()
                                Change.setText("Classname: " + pred_class + " ID: " + SKU_ID)
                                
                                ReportType.setFlags(QtCore.Qt.ItemIsEnabled)
                                TimeStamp.setFlags(QtCore.Qt.ItemIsEnabled)
                                Change.setFlags(QtCore.Qt.ItemIsEnabled)
                                
                                mutex.lock()
                                #Add to report table
                                self.mainwindow.ui.ReportTable.setItem(self.mainwindow.ui.ReportTable.rowCount()-1,0,ReportType)
                                self.mainwindow.ui.ReportTable.setItem(self.mainwindow.ui.ReportTable.rowCount()-1,1,TimeStamp)
                                self.mainwindow.ui.ReportTable.setItem(self.mainwindow.ui.ReportTable.rowCount()-1,2,Change)
                                
                                #Update database table
                                database_tables[sku_id_table_index].table.setItem(sku_id_row_index, quantity_column_index,new_item)  
                                mutex.unlock()  
                                
                        

                            counter+=1
                                           
                             

        
                        
                                            

            if(self.exit==True or runthread==False):

                    print("Camera" + str(self.CameraID) +" Successfully closed") 

                    break     
            
            self.OLD_IMAGE_COORDINATES=detection_output_Person
            self.OLD_IMAGE=img
            self.skipthisiteration=False
            



class CheckModelStartup(QRunnable):
    
    def __init__(self):
        super().__init__()
        self.signals = Signals()
        self.toptext=""
        self.informativetext=""
        self.exitcode=0

    def run(self):
            while(True):
                #Error checking:
                
                #Read settings from file
                with open('settings\setting.txt', 'r') as f:
                    temp = f.readlines()
                
                settings = [item.strip() for item in temp]  
                
                #If database is not setup display error
                
                try:
                    
                    server = settings[0]
                    port=settings[1]
                    database = settings[2]
                    username = settings[3]
                    password = settings[4]


            
                    #Connect to database
                    conn = mysql.connector.connect(user=username, password=password, host=server, database=database,port=int(port))
                    
                except: 
                    self.toptext="Database Error"
                    self.informativetext='Database connection failed. Make sure that the database is properly setup.'
                    self.signals.ErrorSignal.emit(self.toptext,self.informativetext)
                    self.signals.Finished.emit(0)
                    break
                    
                #If settings is not setup display error
                try:
                    temp1=settings[5]
                    temp2=settings[6]
                    temp3=settings[7]
                    temp4=settings[8]
                    temp5=settings[9]
                    temp6=settings[10]
                    temp7=settings[11]
                    
                except:
                    self.toptext="Settings Error"
                    self.informativetext='No saved settings detected. Make sure to save your settings from the settings tab'
                    self.signals.ErrorSignal.emit(self.toptext,self.informativetext)
                    self.signals.Finished.emit(0)
                    break


                #If there is no camera information display error
                try:
                    #Read from file
                    with open('CameraInfo\\Cameras.txt', 'r') as f:
                        temp = f.readlines()
                        
                    lines = [item.strip() for item in temp]  
                    lines[0]
                    
                except:
                    self.toptext="Camera Error"
                    self.informativetext='No saved camera information detected. be sure to properly setup camera information'
                    self.signals.ErrorSignal.emit(self.toptext,self.informativetext)
                    self.signals.Finished.emit(0)
                    break
                
                #Test all camera connections
                try:
                    cameraip=""
                    #Read from file
                    with open('CameraInfo\\Cameras.txt', 'r') as f:
                        temp = f.readlines()
                        
                    lines = [item.strip() for item in temp]  
                    
                        
                    for camerainformation in lines:
                        splitted_cameras=camerainformation.split(" ")
                        cameraip=splitted_cameras[2]
                        CameraType=splitted_cameras[1]
                        
                        if CameraType=="IpWebcam":
                            img_resp = requests.get("http://"+cameraip+":8080//shot.jpg",timeout=10)
                            
                        else:
                            cap = cv2.VideoCapture(int(cameraip))
                            jet,img = cap.read()
                            img = imutils.resize(img, width=1280, height=720)
                    
                except:
                    self.toptext="Camera Error"
                    self.informativetext='Failed to connect to camera with IP=' + cameraip +"\nUpdate the camera information or delete it. Be sure to save your changes."
                    self.signals.ErrorSignal.emit(self.toptext,self.informativetext)
                    self.signals.Finished.emit(0)
                    break
                
                #Check if "items in view" for each camera is setup

                ItemsInView=""
                #Read from file
                with open('CameraInfo\\Cameras.txt', 'r') as f:
                    temp = f.readlines()
                        
                lines = [item.strip() for item in temp]  
                    
                        
                for camerainformation in lines:
                    splitted_cameras=camerainformation.split(" ")
                    ItemsInView=splitted_cameras[3]
                        
                    if ItemsInView=="EMPTY" or ItemsInView=="":
                        self.toptext="Camera Error"
                        self.informativetext='Camera with the ip ' + splitted_cameras[2] +" was never setup.\n Be sure to properly set up the information then save it."
                        self.signals.ErrorSignal.emit(self.toptext,self.informativetext)
                        self.signals.Finished.emit(0)
                        break
                    
                #No error detected return 1   
                self.exitcode=1
                self.signals.Finished.emit(1)  
                break  
                


class AddCameraCheck(QRunnable):
    
    def __init__(self,mainwindowcontext):
        super().__init__()
        self.mainwindow=mainwindowcontext
        self.signals2 = Signals()
        self.exitcode=0
        self.toptext=""
        self.informativetext=""

    def run(self):
        while(True):
            if self.mainwindow.ui.IpWebcamRadio.isChecked():
                url = "http://"+self.mainwindow.ui.CameraIpAddress.text()+":8080//shot.jpg"
                print(url)
                try:
                    img_resp = requests.get(url,timeout=10)
                    self.signals2.Finished.emit(1)
                    self.exitcode=1

                    break
                except:
                    self.toptext="Camera Error"
                    self.informativetext='Failed to add camera with IP =' + self.mainwindow.ui.CameraIpAddress.text()
                    self.signals2.ErrorSignal.emit(self.toptext,self.informativetext)
                    self.signals2.Finished.emit(0)
                    self.exitcode=0

                    break
            elif self.mainwindow.ui.WebcamRadio.isChecked():
                try:
                    #Open camera
                    cap = cv2.VideoCapture(int(self.mainwindow.ui.CameraIpAddress.text()))
                    #Save frame
                    ret,img=cap.read()
                    #Do dummy operation on frame. If the image is null this will throw an exception and tell us that the camera was never opened initially.
                    img = imutils.resize(img, width=1280, height=720)
                    
                    #Send finished signal with code 1 which means a success
                    self.signals2.Finished.emit(1)
                    self.exitcode=1

                    break
                except:
                    self.toptext="Camera Error"
                    self.informativetext='Failed to add camera with Index = ' + self.mainwindow.ui.CameraIpAddress.text()
                    self.signals2.ErrorSignal.emit(self.toptext,self.informativetext)
                    self.signals2.Finished.emit(0)
                    self.exitcode=0

                    break 
            
class SetupDatabaseCheck(QRunnable):
    
        def __init__(self,server,database,username,password,port):
            super().__init__()
            self.server=server
            self.database=database
            self.username=username
            self.password=password
            self.port=port
            
            self.signals3 = Signals()
            self.exitcode=0
            self.toptext=""
            self.informativetext=""
            self.conn=""

        def run(self):
            while(True):

                try:
                    #Connect to database
                    self.conn = mysql.connector.connect(user=self.username, password=self.password, host=self.server, database=self.database,port=int(self.port))
                    cur = self.conn.cursor()
                    self.signals3.Finished.emit(1)
                    self.exitcode=1
                    break
                except:
                    self.toptext="Database Connection Error"
                    self.informativetext="Failed to setup Database. Make sure the connection is open."
                    self.signals3.ErrorSignal.emit(self.toptext,self.informativetext)
                    self.signals3.Finished.emit(0)
                    self.exitcode=0
                    break

class ViewFootageBackground(QRunnable):
    
    def __init__(self,mainwindowcontext):
        super().__init__()
        self.mainwindow=mainwindowcontext
        self.signals4 = Signals()
        self.exitcode=0
        self.toptext=""
        self.informativetext=""

    def run(self):
        
        CameraType=self.mainwindow.ui.CameraTable.item(self.mainwindow.ui.CameraTable.currentRow(), 1).text()
        
        Ipaddress=self.mainwindow.ui.CameraTable.item(self.mainwindow.ui.CameraTable.currentRow(), 2).text()
        
        
        #inport models
        Yolo_model_person_detector = YOLO("Models\\yolov8m.pt")
        YoloModelPath="Models\\YoloModel.pt"  
        YOLO_model = YOLO(YoloModelPath)

                            #Define annotator
        box_annotator = sv.BoxAnnotator(
                                    thickness=1,
                                    text_thickness=1,
                                    text_scale=0.3
                                )

        Label = {
                            0: "Person",
                            1: "bicycle",
                            2: "car",
                            3: "motorcycle",
                            4: "airplane",
                            5: "bus",
                            6: "train",
                            7: "truck",
                            8: "boat",
                            9: "traffic light",
                            10: "fire hydrant",
                            11: "stop sign",
                            12: "parking meter",
                            13: "bench",
                            14: "bird",
                            15: "cat",
                            16: "dog",
                            17: "horse",
                            18: "sheep",
                            19: "cow",
                            20: "elephant",
                            21: "bear",
                            22: "zebra",
                            23: "giraffe",
                            24: "backpack",
                            25: "umbrella",
                            26: "handbag",
                            27: "tie",
                            28: "suitcase",
                            29: "frisbee",
                            30: "skis",
                            31: "snowboard",
                            32: "sports ball",
                            33: "kite",
                            34: "baseball bat",
                            35: "baseball glove",
                            36: "skateboard",
                            37: "surfboard",
                            38: "tennis racket",
                            39: "Product",
                            40: "wine glass",
                            41: "cup",
                            42: "fork",
                            43: "knife",
                            44: "spoon",
                            45: "bowl",
                            46: "banana",
                            47: "apple",
                            48: "sandwich",
                            49: "orange",
                            50: "broccoli",
                            51: "carrot",
                            52: "hot dog",
                            53: "pizza",
                            54: "donut",
                            55: "cake",
                            56: "chair",
                            57: "couch",
                            58: "potted plant",
                            59: "bed",
                            60: "dining table",
                            61: "toilet",
                            62: "tv",
                            63: "laptop",
                            64: "mouse",
                            65: "remote",
                            66: "keyboard",
                            67: "cell phone",
                            68: "microwave",
                            69: "oven",
                            70: "toaster",
                            71: "sink",
                            72: "refrigerator",
                            73: "book",
                            74: "clock",
                            75: "vase",
                            76: "scissors",
                            77: "teddy bear",
                            78: "hair drier",
                            79: "toothbrush"
                                } 
        

        if CameraType=="IpWebcam":
                            url = "http://"+Ipaddress+":8080//shot.jpg"



                            if self.mainwindow.ui.ModelVision.isChecked()==True:
                            
                                                                # While loop to continuously fetching data from the Url
                                                                while True:
                                                                    try:
                                                                        img_resp = requests.get(url)
                                                                        img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
                                                                        img = cv2.imdecode(img_arr, -1)
                                                                        img = imutils.resize(img, width=1000, height=720)
                                                                        
                                                                        detection_output_Person = Yolo_model_person_detector.predict(source=img, conf=0.6, save=False,device='cpu')

                                                                        
                                                                        #Check for a person class in the person model if detected add that information to the result array
                                                                        person_result = detection_output_Person[0]


                                                                        
                                                                        #Convert YOLO results to supervision results
                                                                        detections = sv.Detections.from_yolov8(person_result)
                                                                    
                                                                        empty_detections = sv.Detections.empty()
                                                                        
                                                                        #"Concatenate to model outputs"
                                                                        counter=0
                                                                        for box in person_result.boxes:
                                                                            class_id = box.cls[0].item()
                                                                            if class_id == 39.0 or class_id==0.0:

                                                                                empty_detections.xyxy=np.append(empty_detections.xyxy,values=[(np.array(person_result.boxes[counter].xyxy.tolist(),dtype="float32")).reshape(4,)],axis=0)
                                                                                empty_detections.confidence=np.append(empty_detections.confidence,values=[(np.array(person_result.boxes[counter].conf.tolist(),dtype="float32"))])

                                                                                
                                                                                new_list = [x for x in person_result.boxes[counter].cls.tolist()]
                                                                                empty_detections.class_id=np.append(empty_detections.class_id,values=[(np.array(new_list,dtype='int32'))])
                                                                            counter+=1
                                                                        
                                                                        
                                                                        labels = [f"{Label[class_id]} {confidence:0.2f}"
                                                                                for _, mask,confidence, class_id, tracker
                                                                                in empty_detections
                                                                            ]
                                                                        #Redefine frame
                                                                        frame = box_annotator.annotate(scene=img, detections=empty_detections,labels=labels)

                                                                        
                                                                        #Show frame
                                                                        cv2.imshow("Android_cam", img)
                                                                        self.signals4.Finished.emit(1)
                                                                        # Press Esc key to exit
                                                                        if cv2.waitKey(1) == 27:
                                                                            break
                                                                    except:
                                                                        self.toptext="Camera Error"
                                                                        self.informativetext='Failed to connect to camera with IP =' + Ipaddress
                                                                        self.signals4.ErrorSignal.emit(self.toptext,self.informativetext)
                                                                        self.signals4.Finished.emit(1)
                                                                        break
                                                                
                                                                cv2.destroyAllWindows()
                                                                
                                                                
                            else:
                                
                                                                # While loop to continuously fetching data from the Url
                                                                while True:
                                                                    try:
                                                                        img_resp = requests.get(url)
                                                                        img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
                                                                        img = cv2.imdecode(img_arr, -1)
                                                                        img = imutils.resize(img, width=1000, height=720)
                                                                        

                                                                        
                                                                        #Show frame
                                                                        cv2.imshow("Android_cam", img)
                                                                        self.signals4.Finished.emit(1)
                                                                        # Press Esc key to exit
                                                                        if cv2.waitKey(1) == 27:
                                                                            break
                                                                    except:
                                                                        self.toptext="Camera Error"
                                                                        self.informativetext='Failed to connect to camera with IP =' + Ipaddress
                                                                        self.signals4.ErrorSignal.emit(self.toptext,self.informativetext)
                                                                        self.signals4.Finished.emit(1)
                                                                        break
                                                                
                                                                cv2.destroyAllWindows() 
                                                                
                                                                                               
                                
        if CameraType=="Webcam":   
            
                            #Initalize camera
                            cap = cv2.VideoCapture(int(Ipaddress))
                            cap.set(cv2.CAP_PROP_BUFFERSIZE, 0)
                            cap.set(3, 1280)
                            cap.set(4, 720)
                            if self.mainwindow.ui.ModelVision.isChecked()==True:
                                
                                                                # While loop to continuously fetching data from the Url
                                                                while True:
                                                                    try:
                                                                        
                                                                        ret,img = cap.read()
                                                                        img = imutils.resize(img, width=1280, height=720)
                                                                        
                                                                        #run model on image

                                                                        detection_output_Person = Yolo_model_person_detector.predict(source=img, conf=0.6, save=False,device='cpu')
                                                                        

                                                                        
                                                                        #Check for a person class in the person model if detected add that information to the result array
                                                                        person_result = detection_output_Person[0]


                                                                        
                                                                        #Convert YOLO results to supervision results
                                                                        detections = sv.Detections.from_yolov8(person_result)
                                                                        empty_detections = sv.Detections.empty()
                                                                        
                                                                        #"Concatenate to model outputs"
                                                                        counter=0
                                                                        for box in person_result.boxes:
                                                                            class_id = box.cls[0].item()
                                                                            if class_id == 39.0 or class_id==0.0:

                                                                                empty_detections.xyxy=np.append(empty_detections.xyxy,values=[(np.array(person_result.boxes[counter].xyxy.tolist(),dtype="float32")).reshape(4,)],axis=0)
                                                                                empty_detections.confidence=np.append(empty_detections.confidence,values=[(np.array(person_result.boxes[counter].conf.tolist(),dtype="float32"))])

                                                                                
                                                                                new_list = [x for x in person_result.boxes[counter].cls.tolist()]
                                                                                empty_detections.class_id=np.append(empty_detections.class_id,values=[(np.array(new_list,dtype='int32'))])
                                                                            counter+=1
                                                                        
                                                                        
                                                                        labels = [f"{Label[class_id]} {confidence:0.2f}"
                                                                                for _, mask,confidence, class_id, tracker
                                                                                in empty_detections
                                                                            ]
                                                                        #Redefine frame
                                                                        frame = box_annotator.annotate(scene=img, detections=empty_detections,labels=labels,)

                                                                        
                                                                        #Show frame
                                                                        cv2.imshow("Android_cam", frame)
                                                                        self.signals4.Finished.emit(1)
                                                                        # Press Esc key to exit
                                                                        if cv2.waitKey(1) == 27:
                                                                            break
                                                                    except:
                                                                        self.toptext="Camera Error"
                                                                        self.informativetext='Failed to connect to camera with IP =' + Ipaddress
                                                                        self.signals4.ErrorSignal.emit(self.toptext,self.informativetext)
                                                                        self.signals4.Finished.emit(1)
                                                                        break
                                                                
                                                                cv2.destroyAllWindows()
                                                                
                                                                
                            else:
                                
                                                                # While loop to continuously fetching data from the Url
                                                                while True:
                                                                    try:
                                                                        ret,img = cap.read()
                                                                        img = imutils.resize(img, width=1280, height=720)
                                                                        

                                                                        
                                                                        #Show frame
                                                                        cv2.imshow("Android_cam", img)
                                                                        self.signals4.Finished.emit(1)
                                                                        # Press Esc key to exit
                                                                        if cv2.waitKey(1) == 27:
                                                                            break
                                                                    except:
                                                                        self.toptext="Camera Error"
                                                                        self.informativetext='Failed to connect to camera with IP =' + Ipaddress
                                                                        self.signals4.ErrorSignal.emit(self.toptext,self.informativetext)
                                                                        self.signals4.Finished.emit(1)
                                                                        break
                                                                
                                                                cv2.destroyAllWindows()                      
            
class SetupItemsBackground(QRunnable):
    
    def __init__(self,mainwindowcontext):
        super().__init__()
        self.mainwindow=mainwindowcontext
        self.signals5 = Signals()
        self.exitcode=0
        self.toptext=""
        self.informativetext=""

    def run(self):

        predictions=MODEL.SetupCamera(self.mainwindow)
        #If prediction list is empty i.e a person is in the frame raise an error.
        if len(predictions)==0:
            self.signals5.Finished.emit(0)
            self.toptext="Setup Error"
            self.informativetext="Person detected in the image, Setup exited."
            self.signals5.ErrorSignal.emit(self.toptext,self.informativetext)
        else:
            predictions2=','.join(predictions)
            
            
            ItemsInView = QtWidgets.QTableWidgetItem()
            ItemsInView.setText(predictions2)
            ItemsInView.setFlags(QtCore.Qt.ItemIsEnabled)
            
            self.mainwindow.ui.CameraTable.setItem(self.mainwindow.ui.CameraTable.currentRow(),3,ItemsInView) 
            RuntimeFunctions.SaveCameraTable(self.mainwindow,False)  
            self.signals5.Finished.emit(1)     

        
class SetupItemsCheckCamera(QRunnable):
    
    def __init__(self,mainwindowcontext):
        super().__init__()
        self.mainwindow=mainwindowcontext
        self.signals6 = Signals()
        self.exitcode=0
        self.toptext=""
        self.informativetext=""

    def run(self):
        
        Ipaddress= self.mainwindow.ui.CameraTable.item( self.mainwindow.ui.CameraTable.currentRow(), 2).text()  
        
        if self.mainwindow.ui.CameraTable.item( self.mainwindow.ui.CameraTable.currentRow(), 1).text() == "IpWebcam":
            try: 
                url = "http://"+Ipaddress+":8080//shot.jpg"
                img_resp = requests.get(url,timeout=10)
                img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
                img = cv2.imdecode(img_arr, -1)
                img = imutils.resize(img, width=1280, height=720)
                cv2.imwrite('Models\\YOLOInput\\Yoloimage.jpg',img)
                self.signals6.Finished.emit(1)  
                self.exitcode=1
            except:
                self.toptext="Camera Error"
                self.informativetext='Failed to connect to camera with IP =' + Ipaddress
                self.signals6.ErrorSignal.emit( self.toptext, self.informativetext)
                self.signals6.Finished.emit(0)  
                self.exitcode=0
                
        elif self.mainwindow.ui.CameraTable.item( self.mainwindow.ui.CameraTable.currentRow(), 1).text() == "Webcam":
            try: 
                #Connect to camera
                cap = cv2.VideoCapture(int(Ipaddress))
                cap.set(3, 1280)
                cap.set(4, 720)
                #Get image
                ret,img = cap.read()
                img = imutils.resize(img, width=1280, height=720)
                cv2.imwrite('Models\\YOLOInput\\Yoloimage.jpg',img)
                self.signals6.Finished.emit(1)  
                self.exitcode=1
            except:
                self.toptext="Camera Error"
                self.informativetext='Failed to connect to camera with Index =' + Ipaddress
                self.signals6.ErrorSignal.emit( self.toptext, self.informativetext)
                self.signals6.Finished.emit(0)  
                self.exitcode=0
 
 
 
 
 
 
class SaveDatabaseThread(QRunnable):
    
    def __init__(self,mainwindowcontext):
        super().__init__()
        self.mainwindow=mainwindowcontext
        self.signals7 = Signals()
        self.exitcode=0
        self.toptext=""
        self.informativetext=""

    def run(self):
        while(True):   
            #Read from file
            with open('settings\setting.txt', 'r') as f:
                temp = f.readlines()
                                
            settings = [item.strip() for item in temp]
                                    
                #If database is not setup display error
                    
            try:
                        
                        server = settings[0]
                        port=settings[1]
                        database = settings[2]
                        username = settings[3]
                        password = settings[4]


                
                        #Connect to database
                        conn = mysql.connector.connect(user=username, password=password, host=server, database=database,port=int(port))
                        
            except: 
                        self.toptext="Database Error"
                        self.informativetext='Database connection failed. Make sure that the database is properly setup.'
                        self.signals7.ErrorSignal.emit(self.toptext,self.informativetext)
                        self.signals7.Finished.emit(0)
                        self.exitcode=0
                        break
                        
                    #If settings is not setup display error
            try:
                        temp1=settings[5]
                        temp2=settings[6]
                        temp3=settings[7]
                        temp4=settings[8]
                        temp5=settings[9]
                        temp6=settings[10]
                        temp7=settings[11]
                        
            except:
                        self.toptext="Settings Error"
                        self.informativetext='No saved settings detected. Make sure to save your settings from the settings tab'
                        self.signals7.ErrorSignal.emit(self.toptext,self.informativetext)
                        self.signals7.Finished.emit(0)
                        self.exitcode=0
                        break
            

            #Get QUANTITY column from the database
            cur = conn.cursor()
            cur.execute("SELECT " + settings[9] +  " FROM " + settings[6]+ ";")
            
            QuantityColumn=cur.fetchall()
            

                    
            
            sku_id_table_index=0
            sku_id_column_index=0
            sku_id_row_index=0
            quantity_column_index=0
            quantity_table_index=0
                            
            for i in range(0,self.mainwindow.ui.tabWidget.count()):
                                
                if self.mainwindow.ui.tabWidget.tabText(i)==settings[5]:
                    sku_id_table_index=i
                if self.mainwindow.ui.tabWidget.tabText(i)==settings[6]:
                    quantity_table_index=i
                                    

            for columnindex in range(0,database_tables[quantity_table_index].table.columnCount()):
                                    
                    if database_tables[quantity_table_index].table.horizontalHeaderItem(columnindex).text()==settings[8]:
                        sku_id_column_index=columnindex
                    if database_tables[quantity_table_index].table.horizontalHeaderItem(columnindex).text()==settings[9]:
                        quantity_column_index=columnindex            
            
            
            
            #Read from file
            with open('settings\DB_Quantity.txt', 'r') as f:
                temp = f.readlines()
                                
            DB_QUANTITY = [item.strip() for item in temp]    
                     
            for rowindex in range(0,database_tables[quantity_table_index].table.rowCount()):

                    if int(DB_QUANTITY[rowindex]) - int(database_tables[quantity_table_index].table.item(rowindex, quantity_column_index).text())!=0:
                            cur.execute("Update "+settings[6]+" set "+settings[9]+" = "+ settings[9]  +" - "+ str(int(DB_QUANTITY[rowindex]) - int(database_tables[quantity_table_index].table.item(rowindex, quantity_column_index).text())) +" where "+settings[8]+" = "+database_tables[quantity_table_index].table.item(rowindex, sku_id_column_index).text())   
                             
       
            conn.commit() 
            self.signals7.Finished.emit(1)
            self.exitcode=1
            break
 
 
 
 
class PrintReportBackground(QRunnable):
    
    def __init__(self,mainwindowcontext):
        super().__init__()
        self.mainwindow=mainwindowcontext
        self.signals8 = Signals()
        self.exitcode=0
        self.toptext=""
        self.informativetext=""

    def run(self):
        while(True):
            DataForPDF=[['ReportType','TimeStamp','Change']]
            
            #Get report information and add it to the above array
            for rowindex in range(0,self.mainwindow.ui.ReportTable.rowCount()):
                        row=[]
                        for columnindex in range(0,self.mainwindow.ui.ReportTable.columnCount()):
                            row.append(self.mainwindow.ui.ReportTable.item(rowindex, columnindex).text())
                        DataForPDF.append(row)

            #Initialize PDF object, convert the above data to "TABLE" format then save the the PDF documents with that data.
            elements=[]
            doc=SimpleDocTemplate("Report.pdf",pagesize=A4)   
            table=Table(DataForPDF)
            elements.append(table)
            doc.build(elements)
            self.signals8.Finished.emit(1)
            self.exitcode=1
            break
                              
 
 
 
 
 
            
 
        

class ParallelThreadFunctions():   
    
    def build_model(pretrained=True, fine_tune=True, num_classes=1):
        if pretrained:
            print('[INFO]: Loading pre-trained weights')
        elif not pretrained:
            print('[INFO]: Not loading pre-trained weights')
        model = models.resnet34(pretrained=pretrained)

        if fine_tune:
            print('[INFO]: Fine-tuning all layers...')
            for params in model.parameters():
                params.requires_grad = True
        elif not fine_tune:
            print('[INFO]: Freezing hidden layers...')
            for params in model.parameters():
                params.requires_grad = False
                
        # change the final classification head, it is trainable
        model.fc = nn.Linear(512, num_classes)
        return model

    def StartModel(self):
        
        runnable = CheckModelStartup()
        if self.ui.StartModel.text()=="Start":
            #Display loading icon until check is complete
            self.spinner = QtWaitingSpinner(self, True, True, QtCore.Qt.ApplicationModal)
            self.window().spinner.start()
            
            pool = QThreadPool.globalInstance()
            #Run thread to validate information and display loading icon
            runnable = CheckModelStartup()
            # 3. Call start()
            pool.start(runnable) 
        else:
            ParallelThreadFunctions.modelcheckfinished(ParallelThreadFunctions,1,self)
        
        #Error signal listener. Function will run on error detection
        runnable.signals.ErrorSignal.connect(lambda: ParallelThreadFunctions.DisplayErrorMessage(ParallelThreadFunctions,runnable.toptext,runnable.informativetext))
        
        runnable.signals.Finished.connect(lambda: ParallelThreadFunctions.modelcheckfinished(ParallelThreadFunctions,runnable.exitcode,self))
        

        
                
    def AddCamera(self):

            self.spinner2 = QtWaitingSpinner(self, True, True, QtCore.Qt.ApplicationModal)
            self.window().spinner2.start()
            
            pool = QThreadPool.globalInstance()
            #Run thread to validate information and display loading icon
            runnable = AddCameraCheck(self)
            # 3. Call start()
            pool.start(runnable) 
            
            runnable.signals2.ErrorSignal.connect(lambda: ParallelThreadFunctions.DisplayErrorMessage(ParallelThreadFunctions,runnable.toptext,runnable.informativetext))
            runnable.signals2.Finished.connect(lambda: ParallelThreadFunctions.CheckfinishedActullyAddCamera(ParallelThreadFunctions,runnable.exitcode,self))
            
            
            
            
            
            

    def  SetupItems(self):
        
        for f in os.listdir("Models\\RESNETInput"):
            os.remove(os.path.join("Models\\RESNETInput", f))    
                    
        
        self.spinner5 = QtWaitingSpinner(self, True, True, QtCore.Qt.ApplicationModal)
        self.window().spinner5.start()
        
        pool = QThreadPool.globalInstance()
        #Run thread to validate information and display loading icon
        runnable = SetupItemsCheckCamera(self)
        
        # 3. Call start()
        pool.start(runnable) 
        
        runnable.signals6.Finished.connect(lambda: ParallelThreadFunctions.SetupItemspostprocess(ParallelThreadFunctions,runnable.exitcode,self))
        runnable.signals6.ErrorSignal.connect(lambda: ParallelThreadFunctions.DisplayErrorMessage(ParallelThreadFunctions,runnable.toptext,runnable.informativetext))



        
    
    def  SetupItemspostprocess(self,exitcode,mainwindowcontext):
            if exitcode==1:      
            
                pool = QThreadPool.globalInstance()
                #Run thread to validate information and display loading icon
                runnable = SetupItemsBackground(mainwindowcontext)
                # 3. Call start()
                pool.start(runnable) 

                runnable.signals5.Finished.connect(lambda:  mainwindowcontext.window().spinner5.stop())
                runnable.signals5.ErrorSignal.connect(lambda: ParallelThreadFunctions.DisplayErrorMessage(ParallelThreadFunctions,runnable.toptext,runnable.informativetext))

            else:
                mainwindowcontext.window().spinner5.stop()




    
    
    
    
    
    
        
    def CheckfinishedActullyAddCamera(self,exitcode,mainwindowcontext):
            if exitcode==1: 
                mainwindowcontext.ui.CameraTable.setRowCount(mainwindowcontext.ui.CameraTable.rowCount()+1)                            
                        
                Cam_IP = QtWidgets.QTableWidgetItem()
                Cam_IP.setText(mainwindowcontext.ui.CameraIpAddress.text())
                
                CameraType = QtWidgets.QTableWidgetItem()
                
                
                if mainwindowcontext.ui.IpWebcamRadio.isChecked():
                    CameraType.setText(mainwindowcontext.ui.IpWebcamRadio.text())


                elif mainwindowcontext.ui.WebcamRadio.isChecked():
                    CameraType.setText(mainwindowcontext.ui.WebcamRadio.text())               
        

                
                Cam_ID = QtWidgets.QTableWidgetItem()
                Cam_ID.setText(str(mainwindowcontext.ui.CameraTable.rowCount()))
                
                ItemsInView = QtWidgets.QTableWidgetItem()
                ItemsInView.setText("EMPTY")
                
                Cam_IP.setFlags(QtCore.Qt.ItemIsEnabled)
                CameraType.setFlags(QtCore.Qt.ItemIsEnabled)
                Cam_ID.setFlags(QtCore.Qt.ItemIsEnabled)
                ItemsInView.setFlags(QtCore.Qt.ItemIsEnabled)
                
                mainwindowcontext.ui.CameraTable.setItem(mainwindowcontext.ui.CameraTable.rowCount()-1,0,Cam_ID)
                mainwindowcontext.ui.CameraTable.setItem(mainwindowcontext.ui.CameraTable.rowCount()-1,1,CameraType)
                mainwindowcontext.ui.CameraTable.setItem(mainwindowcontext.ui.CameraTable.rowCount()-1,2,Cam_IP)
                mainwindowcontext.ui.CameraTable.setItem(mainwindowcontext.ui.CameraTable.rowCount()-1,3,ItemsInView)
                
                
                
                
                #Save record to text file
                with open('CameraInfo\\Cameras.txt', 'a') as f:
                    f.write(Cam_ID.text()+" "+ CameraType.text()+ " " + Cam_IP.text() +" " + ItemsInView.text()+"\n")  
                
            mainwindowcontext.window().spinner2.stop()       
        
        
    def modelcheckfinished(self,exitcode,mainwindowcontext):
            global runthread
            
            if exitcode==1:
                #Read from file
                with open('CameraInfo\\Cameras.txt', 'r') as f:
                        temp = f.readlines()
                            
                lines = [item.strip() for item in temp]
                
                
                #Read settings from file
                with open('settings\setting.txt', 'r') as f:
                    temp = f.readlines()
                    
                    settings = [item.strip() for item in temp]  
                            
                            
                if mainwindowcontext.ui.StartModel.text()=="Start":
                    #RUN THREADS
                    runthread=True

                    pool = QThreadPool.globalInstance()
                    for camerainformation in lines:
                        
                        splitted_cameras=camerainformation.split(" ")
                        Itemsinview=splitted_cameras[3]
                        itemsinviewarray=Itemsinview.split(",")
                        # 2. Instantiate the subclass of QRunnable
                        runnable = bkgTask(settings[5],settings[8],settings[6],settings[9],settings[7],settings[10],settings[11],splitted_cameras[2],splitted_cameras[0],mainwindowcontext,splitted_cameras[1],itemsinviewarray)

                        runnable.signals.ErrorSignal.connect(lambda: ParallelThreadFunctions.DisplayErrorMessage(ParallelThreadFunctions,runnable.toptext,runnable.informativetext))
                        
                        runnable.signals.ChangeNotification.connect(lambda: ParallelThreadFunctions.ChangeNotification(self,mainwindowcontext))
                        # 3. Call start()
                        pool.start(runnable)
                        print(pool.activeThreadCount())
        

                    mainwindowcontext.ui.StartModel.setText("Stop")

                
                
                elif mainwindowcontext.ui.StartModel.text()=="Stop":
                    runthread=False 
                    mainwindowcontext.ui.StartModel.setText("Start")  
                    
            else:
                pass
            
            mainwindowcontext.window().spinner.stop()
                  
       
    
        
        
    def ChangeNotification(self,mainwindowcontext):

                            mainwindowcontext.ui.NotificationNumber.setText(str(int(mainwindowcontext.ui.NotificationNumber.text())+1))
                            mainwindowcontext.ui.NotificationFrame.setStyleSheet("background-color: #cc5bce;; border-radius: 12px;")
                            mainwindowcontext.ui.NotificationNumber.setVisible(True)      
                                  
    #This function will be called when the user clicks the setup database button
    def ViewFootage(self):

        

        self.spinner4 = QtWaitingSpinner(self, True, True, QtCore.Qt.ApplicationModal)
        self.window().spinner4.start()

            

        pool = QThreadPool.globalInstance()
        #Run thread to validate information and display loading icon
        runnable = ViewFootageBackground(self)
        # 3. Call start()
        pool.start(runnable) 
            
        runnable.signals4.ErrorSignal.connect(lambda: ParallelThreadFunctions.DisplayErrorMessage(ParallelThreadFunctions,runnable.toptext,runnable.informativetext))
        runnable.signals4.Finished.connect(lambda: self.window().spinner4.stop() )

                    
    
    
    
    
    #This function will be called when the user clicks the setup database button
    def SetupDatabase(self, rowNumber, columnNumber,Getsettingsfromfile):
        global database_tables
        
        self.spinner3 = QtWaitingSpinner(self, True, True, QtCore.Qt.ApplicationModal)
        self.window().spinner3.start()
        
        #Read from file
        with open('settings\setting.txt', 'r') as f:
            temp = f.readlines()
            
            settings = [item.strip() for item in temp]
            
        server=""
        database=""
        username=""
        password=""
        port=""
        
        if Getsettingsfromfile==False:
            server=self.ui.Hostname.text()
            database=self.ui.Databasename.text()
            username=self.ui.Username.text()
            password=self.ui.Password.text()
            port=self.ui.Port.text()
        
        if Getsettingsfromfile==True:
            server = settings[0]
            port=settings[1]
            database = settings[2]
            username = settings[3]
            password = settings[4]
            

        pool = QThreadPool.globalInstance()
        #Run thread to validate information and display loading icon
        runnable = SetupDatabaseCheck(server,database,username,password,port)
        # 3. Call start()
        pool.start(runnable) 
            
        runnable.signals3.ErrorSignal.connect(lambda: ParallelThreadFunctions.DisplayErrorMessage(ParallelThreadFunctions,runnable.toptext,runnable.informativetext))
        runnable.signals3.Finished.connect(lambda: ParallelThreadFunctions.DatabasecheckComplete(ParallelThreadFunctions,runnable.exitcode,self,runnable.conn,Getsettingsfromfile,server,database,username,password,port))    
        
        
    def DatabasecheckComplete(self,exitcode,mainwindowcontext,conn,Getsettingsfromfile,server,database,username,password,port):
        if exitcode==1:
            cur = conn.cursor()
            #Remove preexisting tabs in the table
            try:
                for tabindex in range(0,mainwindowcontext.ui.tabWidget.count()):
                    database_tables[tabindex].verticalLayout.removeWidget(database_tables[tabindex].tab)
                    sip.delete(database_tables[tabindex].tab)
                    database_tables[tabindex].tab = None
                database_tables.clear()
            except:
                print("No tabs to remove")
            

            
            #1. Create a tab for each table in the database

            #Table number
            tablenumber=0
            #Get table names
            cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = '" + database + "'")   
            for table in [tables[0] for tables in cur.fetchall()]:

                #Dynamically create tabs
                tab = QtWidgets.QWidget()
                tab.setObjectName('tab'+str(tablenumber))
                #Create a vertical layout and set it to that tab
                DatabaseverticalLayout = QtWidgets.QVBoxLayout(tab)
                DatabaseverticalLayout.setObjectName('DatabaseverticalLayout' + str(tablenumber))

                #Add that tab to the tabwidget with the name of the table from the database
                mainwindowcontext.ui.tabWidget.addTab(tab,table)
                
                #Now we create the actual table that will be displayed
                cur2 = conn.cursor()
                Databasetable = QtWidgets.QTableWidget(tab)
                Databasetable.setObjectName('Databasetable'+str(tablenumber))
                DatabaseverticalLayout.addWidget(Databasetable)
                
                
                
                #Get number of columns in this table from the database and set it to the table we created
                cur2.execute("SELECT * FROM information_schema.columns WHERE table_name = '" + table+ "';")
                #Save all the column information in a tuple
                columninformation=cur2.fetchall()
                #We need to set the max number of columns to the table and we get that through the length of the tuple since each instance is a column
                columncount=len(columninformation)
                Databasetable.setColumnCount(columncount)
                
                #Add column information to table
                itemsadded=0
                for column in columninformation:
                    item = QtWidgets.QTableWidgetItem()
                    item.setText(QtCore.QCoreApplication.translate("MainWindow", column[3]))
                    Databasetable.setHorizontalHeaderItem(itemsadded, item)
                    itemsadded+=1
                    
                
                #Get number of rows in this table from the database and set it to the table we created
                cur2.execute("SELECT * FROM " + table+ ";")
                #Save all the column information in a tuple
                rowinformation=cur2.fetchall()
                #We need to set the max number of columns to the table and we get that through the length of the tuple since each instance is a column
                rowcount=len(rowinformation)
                Databasetable.setRowCount(rowcount)
                
                #Add row information to table
                itemsadded=0
                for row in rowinformation:
                    for column in range(0, columncount):
                        item = QtWidgets.QTableWidgetItem()
                        item.setText(QtCore.QCoreApplication.translate("MainWindow", str(row[column])))
                        item.setFlags(QtCore.Qt.ItemIsEnabled)
                        Databasetable.setItem(itemsadded,column,item)
                    itemsadded+=1
                    
            
                database_tables.append(DatabaseTables(tab,DatabaseverticalLayout,Databasetable))    
        
                tablenumber+=1 
                
            #Display refresh button
            mainwindowcontext.ui.Refreshdatabase.setVisible(True)
            
            if Getsettingsfromfile==False:
                #Clear file
                with open("settings\setting.txt",'w') as f:
                    pass
                
                #Save record to text file
                with open('settings\setting.txt', 'a') as f:
                    f.write(server+"\n"+ port +"\n" + database+"\n" + username+"\n" + password+"\n")
                
                

            #Setup settings page
            RuntimeFunctions.SetupSettingsPage(mainwindowcontext)  
            RuntimeFunctions.SaveCurrentQuantityValues(mainwindowcontext)
                      
        mainwindowcontext.window().spinner3.stop()       
        
          
    def RefreshDatabase(self):
        
        msgBox = QMessageBox()

        Answer=msgBox.warning(self,'Warning', "Refreshing the database will reset all model updates on it.\nAre you sure you want to refresh??", msgBox.Yes | msgBox.No)
        

        if Answer == msgBox.Yes:
            
            if self.ui.StartModel.text()=="Stop":
                ParallelThreadFunctions.DisplayErrorMessage(self,"Error","Stop model before refeshing database")
                return
            else:
                ParallelThreadFunctions.SetupDatabase(self,0,0,True)    
                
                
    def Savechanges(self):
     
 
        if self.ui.StartModel.text()=="Stop":
            ParallelThreadFunctions.DisplayErrorMessage(self,"Error","Stop model before saving database results")
            return
        
        self.spinner7 = QtWaitingSpinner(self, True, True, QtCore.Qt.ApplicationModal)
        self.window().spinner7.start()   
 
        pool = QThreadPool.globalInstance()
        #Run thread to validate information and display loading icon
        runnable = SaveDatabaseThread(self)
        # 3. Call start()
        pool.start(runnable) 
            
        runnable.signals7.ErrorSignal.connect(lambda: ParallelThreadFunctions.DisplayErrorMessage(ParallelThreadFunctions,runnable.toptext,runnable.informativetext))
        runnable.signals7.Finished.connect(lambda: ParallelThreadFunctions.SavechangedPostProcess(ParallelThreadFunctions,runnable.exitcode,self))            
        
        
        
        
    def SavechangedPostProcess(self,exitcode,mainwindowcontext):       
        if exitcode==1:
            toast=Toast(text='Database updated with current information', duration=3, parent=mainwindowcontext)
            new_point = QPoint(0, 0)
            toast.setPosition(new_point)
            toast.show()
            
        mainwindowcontext.window().spinner7.stop()   
        
        
        
    def ClearReports(self): 
              
        if self.ui.StartModel.text()=="Stop":
            ParallelThreadFunctions.DisplayErrorMessage(self,"Error","Stop model before editing report table")
            return
          
        self.ui.ReportTable.setRowCount(0)  
        
        #Disable undo button
        #Disable the button until another row is clicked
        self.ui.UndoReportChange.setStyleSheet("background-color: rgb(180, 180, 180); border-radius: 8px;")
        self.ui.UndoReportChange.setEnabled(False)   
    def ClearNotifications(self):
        
                if self.ui.StartModel.text()=="Stop":
                    ParallelThreadFunctions.DisplayErrorMessage(self,"Error","Stop model before clearing reports")
                    return
          
                self.ui.NotificationTable.setRowCount(0)  
        
        
    def PrintReport(self):
        if self.ui.StartModel.text()=="Stop":
            ParallelThreadFunctions.DisplayErrorMessage(self,"Error","Stop model before editing report table")
            return
        
        if self.ui.ReportTable.rowCount()==0:
            ParallelThreadFunctions.DisplayErrorMessage(self,"Error","At least 1 record has to be present to print the report")
            return
        
        self.spinner8 = QtWaitingSpinner(self, True, True, QtCore.Qt.ApplicationModal)
        self.window().spinner8.start()   
        
        
        pool = QThreadPool.globalInstance()
        #Run thread to validate information and display loading icon
        runnable = PrintReportBackground(self)
        # 3. Call start()
        pool.start(runnable) 
            
        runnable.signals8.Finished.connect(lambda: ParallelThreadFunctions.PrintReportPostProcess(ParallelThreadFunctions,runnable.exitcode,self))   
        
               
    def PrintReportPostProcess(self,exitcode,mainwindowcontext):       
        if exitcode==1:
            toast=Toast(text='Report Saved', duration=3, parent=mainwindowcontext)
            new_point = QPoint(0, 0)
            toast.setPosition(new_point)
            toast.show()
            os.system("Report.pdf")

            
        mainwindowcontext.window().spinner8.stop()           
        
    def UndoReportChange(self): 
              
        if self.ui.StartModel.text()=="Stop":
            ParallelThreadFunctions.DisplayErrorMessage(self,"Error","Stop model before editing report table")
            return
        
        
        #Get report description from the selected row
          
        Rowselected=self.ui.ReportTable.currentRow()
        
        ReportType=self.ui.ReportTable.item(Rowselected, 0).text()
        Change=self.ui.ReportTable.item(Rowselected, 2).text()
        
        Change_Splitted=Change.split(" ")
        
        
        #Get table name from saved settings

        with open('settings\setting.txt', 'r') as f:
                                temp = f.readlines()
                            
                                settings = [item.strip() for item in temp]
                                
                                
        Quantity_Table_Name=settings[6]
        SKUID_Column_Name=settings[8]
        Quantity_Column_Name=settings[9]
        
        
        quantity_table_index=0
        quantity_column_index=0
        sku_id_column_index=0
        sku_id_row_index=0
        
        
        #Get proper database table indices
        for i in range(0,self.ui.tabWidget.count()):
                                if self.ui.tabWidget.tabText(i)==Quantity_Table_Name:
                                    quantity_table_index=i
                                    break
                                    

        for columnindex in range(0,database_tables[quantity_table_index].table.columnCount()):
                                    
                                    if database_tables[quantity_table_index].table.horizontalHeaderItem(columnindex).text()==SKUID_Column_Name:
                                        sku_id_column_index=columnindex
                                    if database_tables[quantity_table_index].table.horizontalHeaderItem(columnindex).text()==Quantity_Column_Name:
                                        quantity_column_index=columnindex   
                                        
        for rowindex in range(0,database_tables[quantity_table_index].table.rowCount()):

                                    if database_tables[quantity_table_index].table.item(rowindex, sku_id_column_index).text()==Change_Splitted[3]:
                                        sku_id_row_index= rowindex    
                                        break
                                        
        #If report type is subtraction then add to inital database table and vice versa
        if ReportType=="Product Subtraction":  
            
            new_item = QtWidgets.QTableWidgetItem()
            new_item.setText(str(int(database_tables[quantity_table_index].table.item(sku_id_row_index, quantity_column_index).text())+1))  
            new_item.setFlags(QtCore.Qt.ItemIsEnabled)  
            #Reverse the process and update database table
            database_tables[quantity_table_index].table.setItem(sku_id_row_index, quantity_column_index,new_item) 
            
        if ReportType=="Product Addition":  
            
            new_item = QtWidgets.QTableWidgetItem()
            new_item.setText(str(int(database_tables[quantity_table_index].table.item(sku_id_row_index, quantity_column_index).text())-1))  
            new_item.setFlags(QtCore.Qt.ItemIsEnabled)  
            #Reverse the process and update database table
            database_tables[quantity_table_index].table.setItem(sku_id_row_index, quantity_column_index,new_item) 
            
        #Delete the row   
        self.ui.ReportTable.removeRow(self.ui.ReportTable.currentRow())

        #Disable the button until another row is clicked
        self.ui.UndoReportChange.setStyleSheet("background-color: rgb(180, 180, 180); border-radius: 8px;")
        self.ui.UndoReportChange.setEnabled(False)   
            
            
            
            
            
            
            
            
    def GetCellClickedReportTable(self,row):
        
        self.ui.UndoReportChange.setStyleSheet("background-color:#cc5bce; border-radius: 8px;")
        self.ui.UndoReportChange.setEnabled(True)             
                   
                                        
    def DisplayErrorMessage(self, toptext,informativetext):
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Critical)
                msg.setText(toptext)
                msg.setInformativeText(informativetext)
                msg.setWindowTitle("Error")
                msg.exec_()
                            
                