import cv2
import json
import numpy as np
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from tensorflow.keras.models import load_model

model_path="./model.h5"
model=load_model(model_path)

with open("labels_dict.json") as f:
    labe_dict=json.load(f)
final_dict={}
for disease,idx in labe_dict.items():
    final_dict[idx]=disease

test_image_path=r"./Dataset/test/Tomato___Bacterial_spot (1).JPG"\

tomato_plant=cv2.imread(test_image_path)
test_image=cv2.resize(tomato_plant,(128,128))

#==========Convert image to numpy array and normalize==========
test_image=img_to_array(test_image)/255
#=======================change dimention 3D to 4D==============
test_image=np.expand_dims(test_image,axis=0)

result=model.predict(test_image)

pred=np.argmax(result,axis=1)

predicted_val=int(pred)

predicted_disease=final_dict[predicted_val]
print(predicted_disease)
