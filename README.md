# KAIM
KAIM is a retail inventory mangement system that utilizes AI and cameras to provide an all encompassing inventory management system. This project was our CS Bachelor degree graduation project. The project was supervised by [Dr. Bander AlSulami](https://sa.linkedin.com/in/bander-alsulami-ph-d-6011a8a).

This document will not cover everything, only the main points. If you would like a more indepth look, refer to the KAIM_FINALREPORT Document.


# DEMO:

In this example the clorox bottle was subtracted from the database then added, while skipping frames where the human was visible.






https://github.com/Kalal0/Retail-Smart-Inventory-Management-based-on-Real-time-Analytics/assets/109832303/f9ed2fb1-89a9-4207-87cb-7089feba7988




Our system works by comparing the current frame with the previous frame, if it detects a change it will update the database accordingly either by subtraction or addition. 

To ensure no false updates take place, the system will skip any frames that contain humans. 
This could be further imporved to include other objects such as shopping carts and handbags. (i.e common items that could block the camera's vision and lead to false updates)




# Team members: 
  - [Khalid Alghamdy](https://github.com/Kalal0)
  - Abdulrahman Alalyan 441102443@student.ksu.edu.sa
  - [Ibrahim Alazba](https://github.com/ibrahim-alazba)
  - [Mohanned Alghonaim](https://github.com/Kokuten7777)



# System overview
The system is comprised of four parts: 
  - Camera
  - Server
  - Database
  - Product Shelf

The server is connected to both the store database and all the cameras that are monitoring the store shelves. Whenever a new product is added or taken from that shelf, our AI model will detect that change and send the appropriate information to the server. From there the server will update the store database accordingly.

Below is the bare bone structure of our system: 

![262988529-34991126-093d-4dfb-ada0-0eeaea5a197e](https://github.com/Kalal0/Retail-Smart-Inventory-Management-based-on-Real-time-Analytics/assets/109832303/c59d754c-1037-41b3-9fca-8f0e64f367e6)


# How did we achieve this?
The problem was split up into 2 parts:
  - Where is the product located?
  - What is the product?

Each of these two problems required a different approach to solve. 

## The "Where"
  To know where the products are located, we used a [YOLO](https://github.com/ultralytics/ultralytics) machine learning model and trained it on the [SKU110K dataset](https://github.com/eg4000/SKU110K_CVPR19).

![263541299-10e7a644-c1f9-4bbf-a8fc-2d2f2b1aba5e](https://github.com/Kalal0/Retail-Smart-Inventory-Management-based-on-Real-time-Analytics/assets/109832303/4e82d0e7-7ce1-46e1-b05e-5ab64e806391)



## The "What"
  Seeing as how there isn't any publicly available "fine grained" datasets for Saudi Arabian retailing store products, we had to create our own. We gathered our dataset by taking pictures of store shelves and passing those images to our YOLO model to provide us with indivdual product images. The stores were Al-Danube and Al-Tamimi. After structuring our dataset we trained it using a [Resnet34](https://www.kaggle.com/datasets/pytorch/resnet34) Model.

#### Dataset gathering process: 

![263541359-5ec1c740-c37b-4466-a380-7a4e5e07c117](https://github.com/Kalal0/Retail-Smart-Inventory-Management-based-on-Real-time-Analytics/assets/109832303/a961ea3f-2bd5-478e-b86c-4c736f734cfc)



# Linking it with the store database
  The program has a GUI interface, the user has the option to add the database information (Database IP, Name, Password, etc... ). The Program will then pull all the table information from the database and display them in the program.

# Main program loop:
  After the camera(s) is/are connected, and the database is linked. Some other settings should be set up beforehand such as snapshot frequency. The default is 10 secs, this means that the system will compare camera frames every 10 seconds and search for a change. The amount of cameras that can be connected at once depends on the number of cores the server CPU has. Finally the user can start the main program loop by a click of a button.


# Screenshots

### Login Screen (user/pass=0000)

![263540791-535a048f-7dcb-45e5-9ea5-dc8815a11e58](https://github.com/Kalal0/Retail-Smart-Inventory-Management-based-on-Real-time-Analytics/assets/109832303/11dad8cc-813e-42eb-9aff-1e0d0a514137)


### Database Viewer

![263540835-7fb78c73-95d0-4b75-848e-4995ce39c4ac](https://github.com/Kalal0/Retail-Smart-Inventory-Management-based-on-Real-time-Analytics/assets/109832303/81912337-8d0a-483a-b2c6-d6cd83ea1dcf)


### Database Viewer after connecting to an external database
![263540883-df6e5bd6-68ea-47da-9ac3-d0e9b61d0f7a](https://github.com/Kalal0/Retail-Smart-Inventory-Management-based-on-Real-time-Analytics/assets/109832303/e48c432f-7f0d-45ae-98bf-c9a65ef057e3)



### Settings tab
![263540907-2c7fa8e8-3c7e-43d8-8a8e-cfa3be778ba1](https://github.com/Kalal0/Retail-Smart-Inventory-Management-based-on-Real-time-Analytics/assets/109832303/5a3968e1-500b-41c4-b677-864e60562d97)



### Main Program Flowchart 


![263543709-ae596651-a127-4171-ac93-6382db803a57](https://github.com/Kalal0/Retail-Smart-Inventory-Management-based-on-Real-time-Analytics/assets/109832303/a8e8c4c4-37a9-408b-a2a2-32bb7e4f83f0)



