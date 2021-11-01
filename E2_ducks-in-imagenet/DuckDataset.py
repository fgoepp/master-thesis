import torch
import torchvision.transforms as transforms

import numpy as np

from PIL import Image, ImageFile

class DuckDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 background_dataset, 
                 duck_img = 'duck.png', # our beloved duck
                 uniform_yellow = False,
                 p_duck = 0.5,
                 random_horizontal_flip = True,
                 random_rotation = (-180, 180),
                 random_size = (80, 80),
                 transform = None, # transformation to apply after we placed the duck,
                 length = None,
            ):
            super(DuckDataset, self).__init__()
            
            if length is None:
                length = len(background_dataset)
            assert length <= len(background_dataset), 'DuckDataset: length of the background dataset is smaller than the specified length'
            
            self.background_dataset = background_dataset
            self.p_duck = p_duck
            self.uniform_yellow = uniform_yellow 
            self.random_horizontal_flip = random_horizontal_flip
            self.random_rotation = random_rotation
            self.random_size = random_size
            self.duck_img = Image.open(duck_img) 
            self.duck_img.load() # this is necessary in order to work with multiprocessing, i.e. multiple dataloaders
            self.duck_img = self.duck_img.resize((128, 128))
            self.transform = transform
            self.length = length
            

    def __getitem__(self, index):
        img, label, _ , _ = self.__draw__(index)
        return img, label
    
    def __draw_random__(self):
        return self.__draw__(np.random.randint(self.length))
    
    def __draw__(self, index):
        assert index < self.length, 'Invalid index!'
    
        # get the background image
        img, _ = self.background_dataset[index]
        bgr = img.copy()
        # do we have a duck?
        label = int(np.random.random() < self.p_duck)
        # optionally, the location of the duck
        duck_location = None 
        # then place the duck!
        if label == 1:
            duck_transformed = self.duck_img
            
            # optional horizontal flip
            if self.random_horizontal_flip:
                duck_transformed = transforms.RandomHorizontalFlip()(duck_transformed)
            # size of the duck
            if self.random_size[0] < self.random_size[1]:
                size = np.random.randint(self.random_size[0], self.random_size[1])
            else:
                size = self.random_size[0]
            # apply transform
            duck_transform = transforms.Compose([transforms.RandomRotation(self.random_rotation, expand=True),
                                                 transforms.Resize((size, size))])
            duck_transformed = duck_transform(duck_transformed)
            # random position
            x = np.random.randint(img.width-size)
            y = np.random.randint(img.height-size)
            duck_location = (x, y, size)
            img.paste(duck_transformed, (x, y), duck_transformed)
        # apply transform to final image
        if self.transform is not None:
            img = self.transform(img)
            bgr = self.transform(bgr)
            
            if self.uniform_yellow:
                mask = img - bgr

                mask[mask != 0] = 1
                mask[:] = mask.sum(axis = 0)

                img[2][mask[2] != 0] = 0 
                img[0:2][mask[0:2] != 0] = 1
                
        return img, label, duck_location, bgr
            
    def __len__(self):
        return self.length