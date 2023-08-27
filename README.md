# KAIM
KAIM is a retail inventory mangement system that utilizes AI and cameras to provide an all encompassing inventory management system. This project was our CS Bachelor degree graduation project. The project was supervised by [Dr. Bander AlSulami](https://sa.linkedin.com/in/bander-alsulami-ph-d-6011a8a).

# Team members: 
  - Khalid Alghamdy 441100858@student.ksu.edu.sa
  - Abdulrahman Alalyan 441102443@student.ksu.edu.sa
  - Ibrahim Alathbah 441103443@student.ksu.edu.sa
  - Mohanned Alghonaim 441101453@student.ksu.edu.sa



# System overview
The system is comprised of four parts: 
  - Camera
  - Server
  - Database
  - Product Shelf

The server is connected to both the store database and all the cameras that are monitoring the store shelves. Whenever a new product is added or taken from that shelf, our AI model will detect that change and send the appropriate information to the server. From there the server will update the store database accordingly.

Below is the bare bone structure of our system: 

![System Architecture](https://github.com/Kalal0/Retail-Smart-Inventory-Management-based-on-Real-time-Analytics/assets/109832303/34991126-093d-4dfb-ada0-0eeaea5a197e)

# How did we achieve this?
The problem was split up into 2 parts:
  - Where is the product located?
  - What is the product?

Each of these two problems required a different approach to solve. 

## The "Where"
  To know where the products are located, we used a [YOLO](https://github.com/ultralytics/ultralytics) machine learning model and trained it on the [SKU110K dataset](https://github.com/eg4000/SKU110K_CVPR19).

## The "What"
  Seeing as how there isn't any publicly available "fine grained" datasets for Saudi Arabian retailing store products, we had to create our own. We gathered our dataset by taking pictures of store shelves and passing those images to our YOLO model to provide us with indivdual product images. The stores were Al-Danube and Al-Tamimi. After structuring our dataset we trained it using a [Resnet34](https://www.kaggle.com/datasets/pytorch/resnet34) Model.


# Linking it with the store database
  The program has a GUI interface, the user has the option to add the database information (Database IP, Name, Password, etc... ). The Program will then pull all the table information from the database and display them in the program.

# Main program loop:
  After the camera(s) is/are connected, and the database is linked. Some other settings should be set up beforehand such as snapshot frequency. The default is 10 secs, this means that the system will compare camera frames every 10 seconds and search for a change. The amount of cameras that can be connected at once depends on the number of cores the server CPU has. Finally the user can start the main program loop by a click of a button.
    
