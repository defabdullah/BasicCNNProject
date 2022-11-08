import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import normalize_rgb

def predict_cat_dog(model,test_dir = "test1/", number=1 ,image_size=256):
    test = cv2.imread(os.path.join(test_dir,str(number)+".jpg"))
    if test is not None:
        plt.imshow(test)
        test = cv2.resize(test,(image_size,image_size))
        test = test.astype(np.float32)
        test = normalize_rgb(test)
        test = np.expand_dims(test, axis=0)
        prediction = model.predict(test, batch_size=1)
        if(prediction > 0.5):
            result = "CAT"
        else:
            result=  "DOG"
        print(prediction)
    else:
        print("Image is None")