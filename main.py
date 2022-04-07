import torch
from model import SpinalVGG
from segmentation import Segmentation
import cv2 as cv2
import numpy as np
import os

device = 'cuda'
model1 = SpinalVGG().to(device)
model1.load_state_dict(torch.load("WEIGHT\\model.h5"))

test_folder = "TEST_CAPTCHAS"
for i in os.listdir(test_folder):
    path = f"{test_folder}\\{i}"
    path_list = Segmentation(path)
    classes = "A,C,E,H,N,P,R,S,T,U,W,X,Y,Z,b,d,n,q,t,1,2,3,4,5,6,7"
    index = "10,12,14,17,23,25,27,28,29,30,32,33,34,35,37,38,43,44,46,62,63,64,65,66,67,68"
    values = classes.split(',')
    keys = index.split(',')
    output_dict = dict(zip(keys, values))
    ans_list = []
    for j in path_list:
        img = cv2.imread(j, 0)
        img = cv2.resize(img, (28, 28))
        img = np.reshape(img, (1, 1, 28, 28))
        img = torch.tensor(img, dtype=torch.float)
        img = img.to(device)
        model1.eval()
        x = model1(img)
        _, predicted = torch.max(x.data, 1)
        ans_list.append(output_dict[f'{predicted.item()}'])

    ans_list = ''.join(map(str, ans_list))
    print(ans_list)
