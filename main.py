# https://stackoverflow.com/questions/60116208/pytorch-load-dataset-of-grayscale-images
import torch
import torchvision


import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os

from PIL import Image
import copy
import torch.nn.functional as F


from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from torchvision import transforms



from PIL import Image
import requests
from io import BytesIO
import streamlit as st
import validators



label_map={
    0:"Chickenpox",
    1:"Measles",
    2:"Monkeypox",
    3:"Normal"
}
classes = ('Chickenpox', 'Measles', 'Monkeypox', 'Normal')
PATH = './resnet18_net.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                     transforms.Resize((64,64)),
                                     transforms.ToTensor()])


def load_model():
	'''
	load a model 
	by default it is resnet 18 for now
	'''
	model = models.resnet18(pretrained=True)
	num_ftrs = model.fc.in_features
	model.fc = nn.Linear(num_ftrs, len(classes))
	model.to(device)

	model.load_state_dict(torch.load(PATH,map_location=device))
	model.eval()
	return model




def predict(model, image_url):
	'''
	pass the model and image url to the function
	Returns: a list of pox types with decreasing probability
	'''
	if validators.url(image_url) is True:	
		response = requests.get(image_url)		
		picture = Image.open(BytesIO(response.content))
	else:
		picture = Image.open(image_url)
	# Convert the image to grayscale and other transforms
	image = data_transform(picture)
	# store in a list of images
	images=image.reshape(1,1,64,64)
	new_images = images.repeat(1, 3, 1, 1)
	outputs=model(new_images)
	# get prediction
	_, predicted = torch.max(outputs, 1)
	ranked_labels=torch.argsort(outputs,1)[0]
	# get all classes in order of probability
	probable_classes=[]
	for label in ranked_labels:
	    probable_classes.append(classes[label.numpy()])
	probable_classes.reverse()
	return probable_classes







if __name__=="__main__":
	model=load_model()
	print("Model loaded")
	# normal
	image_url="https://drive.google.com/uc?export=view&id=14sF_FaFvfYzrQCCQRX6IK87aBPFerfWb"	
	print(predict(model, image_url),"should be normal")

	image_url="/Users/ashhadulislam/projects/other_misc/hackerEarth/daisi/some_pox_data/Monkeypox-dataset-2022-master/arranged/Normal/normalgray_aug14.jpg"	
	print(predict(model, image_url),"should be normal")	

	# chicken pox
	image_url="https://drive.google.com/uc?export=view&id=1nwBZQb0R0L4TMuk_9PaG9Hh2LhWRYz5R"	
	print(predict(model, image_url),"should be chicken pox")


	# measles
	image_url="https://drive.google.com/uc?export=view&id=1GFkAo0LFARMd9pfKV3iY973T8iZ6qyYq"	
	print(predict(model, image_url),"should be measles")

	# monkey pox
	image_url="https://drive.google.com/uc?export=view&id=1Fg9mhVjnsMHKKKHzgNDG3IRhWapfufbw"	
	print(predict(model, image_url),"should be monkeypox")



	print("Going for color images")
	# trying color images
	# normal
	image_url="https://drive.google.com/uc?export=view&id=1kLVQi8O4OIvkrvfJd8zTAVHPHrVEgntY"	
	print(predict(model, image_url),"should be normal")

	# chicken pox
	image_url="https://drive.google.com/uc?export=view&id=1Fptf7tOEz6y_rG4NEpNzWyrjEO5NqURg"	
	print(predict(model, image_url),"should be chicken pox")


	# measles
	image_url="https://drive.google.com/uc?export=view&id=1UDXP3rhFU9VZF-fXV36YPAzUY419tWfW"	
	print(predict(model, image_url),"should be measles")

	# monkey pox
	image_url="https://drive.google.com/uc?export=view&id=1MaKJYC1RJdxk9rWNxcdXlgtCE2eLsSGy"	
	print(predict(model, image_url),"should be monkeypox")


	print("Going for local images")
	image_url="/Users/ashhadulislam/projects/other_misc/hackerEarth/daisi/some_pox_data/Monkeypox-dataset-2022-master/arranged/Chickenpox/chicken21.jpg"
	print(predict(model, image_url),"should be chickenpox")	


	

















