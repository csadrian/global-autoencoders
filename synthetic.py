import gin
import gin.torch
import gin.torch.external_configurables

#generate synthetic data
import random
import math
import numpy as np
import torch

#import matplotlib.pyplot as plt
@gin.configurable('Flower')
class Flower(torch.utils.data.Dataset):
    def __init__(self, train, n_points=50000,
                 petals=50, petal_length= .5, petal_width=.003, seed = 0):
        if train == True:
            self.data = flower(n_points, petals, petal_length, petal_width, seed)
        else:
            self.data = flower(int(n_points/4), petals, petal_length, petal_width, seed+1)
        self.labels = torch.full((len(self.data),), 0, dtype=torch.int)
            

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.data[idx,:], self.labels[idx]
        

def angletocoords2d(angle, radius):
    return radius * np.array([math.cos(angle), math.sin(angle)])

def flower(n_points, petals, petal_length, petal_width, seed):
    #plant the flower seed
    random.seed(seed)
    flowerpoints = np.zeros([n_points, 2])
    for p in range(n_points):
        petal = random.randint(0, petals)
        #find center
        if petal == 0:
            center = np.array([.5, .5])
        else:
            angle0 = (petal - 1) / (petals - 1) * 2 * math.pi
            radius0 = petal_length * random.random()
            center = np.array([.5, .5]) + angletocoords2d(angle0, radius0)

        angle1 = 2 * math.pi * random.random()
        radius1 = petal_width * random.random() * (.75 + -abs(radius0 / petal_length - .75))
        if petal == 0:
            radius1 *= 2
        translation = angletocoords2d(angle1, radius1)
        flowerpoints[p,:] = center + translation
    return torch.tensor(flowerpoints, dtype = torch.float)

#flowerpoints = flower(10000, 12, 8., 0.2)
#plt.scatter(flowerpoints[:,0], flowerpoints[:,1])
#plt.show()
