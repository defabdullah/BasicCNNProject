import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def generate_batches(dataParser, train=True):
    while True:
        if train:
            batch_ids = np.random.choice(dataParser.training_ids, dataParser.batch_size, replace=False)
        else:
            batch_ids = np.random.choice(dataParser.validation_ids, dataParser.batch_size*2, replace=False)
        images, labels = dataParser.get_batch(batch_ids)
        yield(images, labels)

def label_files_with_names(train_path):
    file_names=[]
    for file_name in os.listdir(train_path):
        result=0
        if "cat" in file_name:
            result=1
        file_names.append((os.path.join(train_path,file_name), result))
    return file_names

def normalize_rgb(img):
    im = img.shape
    img = img.reshape((img.shape[0]*img.shape[1]), img.shape[2])
    scaler = MinMaxScaler()
    scaler = scaler.fit(img)
    img = scaler.fit_transform(img)
    img = img.reshape(im[0], im[1], im[2])
    return img