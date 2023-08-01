#Inports
from PIL import Image 
import torchvision.models as models
import torch.nn as nn
import torch
import cv2
import torchvision.transforms as transforms
import argparse
from IPython import display
import ultralytics
from ultralytics import YOLO
import os
import supervision as sv







class MODEL():
    
    
    
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
    
    def SetupCamera(self):
        ##YOLO
        YoloModelPath="Models\\YoloModel.pt"
        
        ##YOLO_PERSON_DETECTOR
        Yolo_model_person_detector = YOLO("Models\\yolov8m.pt")

        #initialize YOLO model: 
        YOLO_model = YOLO(YoloModelPath)
        
        detection_output_Person = Yolo_model_person_detector.predict(source="Models\\YOLOInput\\Yoloimage.jpg", conf=0.6, save=False,device='cpu')
        
        
        #Check if a person is in the image. If there is then disgard entire frame and return an empty list to then raise an exception in the calling method.
        person_result = detection_output_Person[0]
        
        for box in person_result.boxes:
            class_id = box.cls[0].item()
            #If theres a bounding box  the class 0 which is the person class then exit method
            if class_id == 0.0:
                return []

        #Crop all bounding boxes and save results
        counter2=1
        #Iterate through the results
        for box in person_result.boxes:
                class_id = box.cls[0].item()
                if class_id == 39.0:
                    #This object now contains the xy coordinates of 1 bounding box
                    cords = box.xyxy[0].tolist()
                    
                
            

                    #Get topleft and bottom right coordinates                   
                    x1 = cords[0]
                    y1 = cords[1]
                    x2 = cords[2]
                    y2 = cords[3]
                    
                    #Crop image then save
                    img = Image.open("Models\\YOLOInput\\Yoloimage.jpg") 
                    img_res = img.crop((x1, y1, x2, y2)) 
                    img_res.save("Models\\RESNETInput\\"+"Output"+str(counter2)+".jpg")
                    counter2+=1
        
        ##RESNET
        
        with open('Models\\Classnames.txt', 'r') as f:
            Classnames = [line.strip() for line in f]
            
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
        
        predictions= []
        
        #Open files
        images = os.listdir("Models\\RESNETInput")
        
        for image in images:
                    imagepath="Models\\RESNETInput\\"+image
                    # read and preprocess the image
                    image = cv2.imread(imagepath)
                    orig_image = image.copy()
                    # convert to RGB format
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image = transform(image)
                    # add batch dimension
                    image = torch.unsqueeze(image, 0)
                    with torch.no_grad():
                        outputs = RESNET_model(image.to(device))
                    output_label = torch.topk(outputs, 1)
                    pred_class = Classnames[int(output_label.indices)]  
                    if pred_class not in predictions:
                        predictions.append(pred_class)
                        
        if len(predictions)==0:
            predictions.append("EMPTY")
        return predictions         

