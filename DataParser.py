import numpy as np
import cv2
from utils import label_files_with_names,normalize_rgb

class DataParser():

    def __init__(self, train_path, validation_split=0.3, batch_size=16, image_size=256,rate=1):
        
        self.samples = label_files_with_names(train_path)
        self.n_samples = len(self.samples)
        self.all_ids = list(range(self.n_samples))
        
        np.random.shuffle(self.all_ids)
        train_split = 1 - validation_split
        self.training_ids = self.all_ids[:int(train_split * self.n_samples*rate)]
        self.validation_ids = self.all_ids[int(train_split*self.n_samples*rate):int(train_split * self.n_samples*rate)+int(validation_split * self.n_samples*rate)]

        self.batch_size = batch_size
        self.steps_per_epoch = len(self.training_ids)/batch_size
        self.validation_steps = len(self.validation_ids)/(batch_size*2)
        self.image_size = image_size
    
    def get_batch(self, batch):
        images = np.empty([len(batch),self.image_size,self.image_size,3])
        labels = np.empty([len(batch),1])
        for i in range(len(batch)):
            b = batch[i]
            im = cv2.imread(self.samples[b][0])
            if im is None:
                continue
            im = cv2.resize(im,(self.image_size,self.image_size))
            im = im.astype(np.float32)
            im = normalize_rgb(im)
            images[i] = im
            labels[i] = self.samples[b][1]
        return images, labels