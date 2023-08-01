import torch
import cv2
import torchvision.transforms as transforms
import argparse

from model import build_model

# construct the argument parser
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', 
    default='input/test_data/cloudy.jpeg',
    help='path to the input image')
args = vars(parser.parse_args())

# the computation device
device = 'cpu'

# list containing all the labels
with open('Classnames.txt', 'r') as f:
    Classnames = [line.strip() for line in f]

# initialize the model and load the trained weights
model = build_model(
    pretrained=False, fine_tune=False, num_classes=len(Classnames)
).to(device)
print('[INFO]: Loading custom-trained weights...')
checkpoint = torch.load('outputs/model.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

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

# read and preprocess the image
image = cv2.imread(args['input'])
# get the ground truth class
gt_class = args['input'].split('/')[-1].split('.')[0]
orig_image = image.copy()
# convert to RGB format
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = transform(image)
# add batch dimension
image = torch.unsqueeze(image, 0)
with torch.no_grad():
    outputs = model(image.to(device))
output_label = torch.topk(outputs, 1)
pred_class = Classnames[int(output_label.indices)]
cv2.putText(orig_image, 
    f"GT: {gt_class}",
    (10, 25),
    cv2.FONT_HERSHEY_SIMPLEX, 
    1, (0, 255, 0), 2, cv2.LINE_AA
)
cv2.putText(orig_image, 
    f"Pred: {pred_class}",
    (10, 55),
    cv2.FONT_HERSHEY_SIMPLEX, 
    1, (0, 0, 255), 2, cv2.LINE_AA
)
print(f"GT: {gt_class}, pred: {pred_class}")
cv2.imshow('Result', orig_image)
cv2.waitKey(0)
cv2.imwrite(f"outputs/{gt_class}.png",
    orig_image)
