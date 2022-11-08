import os
import numpy as np
from sklearn.preprocessing import StandardScaler

def generate_batches(dataParser, train=True):
    while True:
        if train:
            batch_ids = np.random.choice(dataParser.training_ids, dataParser.batch_size, replace=False)
        else:
            batch_ids = np.random.choice(dataParser.validation_ids, dataParser.batch_size, replace=False)
        images, labels = dataParser.get_batch(batch_ids)
        yield(images, labels)

def label_files_with_names(train_path):
    file_names=[]
    for file_name in os.listdir(train_path):
        result=0
        if "cat" in file_name:
            result=1
        elif "dog" not in file_name:
            continue
        file_names.append((os.path.join(train_path,file_name), result))
    return file_names

def normalize_rgb(img):
    im = img
    img = img.reshape((img.shape[0]*img.shape[1]), img.shape[2])
    scaler = StandardScaler()
    scaler = scaler.fit(img)
    img = scaler.fit_transform(img)
    img = img.reshape(im.shape[0], im.shape[1], im.shape[2])
    return img