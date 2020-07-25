import gin
import gin.torch
import gin.torch.external_configurables

#generate synthetic data
import random
import math
import numpy as np
import torch

import matplotlib.pyplot as plt

@gin.configurable('SquareGrid')
class SquareGrid(torch.utils.data.Dataset):
    def __init__(self, train, n_points=20000, gridlines = 125, seed = 0):
        self.gridlines = 30
        if train == True:
            self.data, self.labels  = grid(n_points, gridlines, seed)
        else:
            self.data, self.labels  = grid(int(n_points /4), gridlines, seed + 1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.data[idx,:], self.labels[idx]

#to avoid multiple points in 0, let n_lines divide n_points (simple implementation)
def grid(n_points, n_lines, seed):
    random.seed(seed)
    n_pointsonaline = int(n_points / n_lines)
    grid_points = np.zeros([n_points, 2])
    grid_labels = np.zeros([n_points])
    for x in range(n_lines):
        for y in range(n_pointsonaline):
            angle1 = 2 * math.pi * random.random()
            radius1 =  random.random() / n_lines

            grid_points[x * n_pointsonaline + y,:] = np.array([x / n_lines,  y / n_pointsonaline]) + angletocoords2d(angle1, radius1) / n_pointsonaline
    return torch.tensor(grid_points, dtype = torch.float), torch.tensor(grid_labels, dtype = torch.int)


@gin.configurable('Flower')
class Flower(torch.utils.data.Dataset):
    def __init__(self, train, n_points=50000,
                 petals=50, petal_length= .5, petal_width=.003, seed = 0):
        self.petals = petals
        if train == True:
            self.data, self.labels = flower(n_points, petals, petal_length, petal_width, seed)
        else:
            self.data, self.labels  = flower(int(n_points/4), petals, petal_length, petal_width, seed+1)
        #self.labels = torch.full((len(self.data),), 0, dtype=torch.int)
            

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.data[idx,:], self.labels[idx]

@gin.configurable('GaussFlower')
class Flower(torch.utils.data.Dataset):
    def __init__(self, train, n_points=50000,
                 petals=10, seed = 0):
        self.petals = petals
        if train == True:
            self.data = flower(n_points, petals, seed)
        else:
            self.data = flower(int(n_points/4), petals, seed+1)
        #self.labels = torch.full((len(self.data),), 0, dtype=torch.int)
            

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.data[idx,:], self.labels[idx]

    
@gin.configurable('Snail')
class Snail(torch.utils.data.Dataset):
    def __init__(self, train, n_points=50000,
                 bend = 2, width=.05, dim = 2, seed = 0):
        if train == True:
            self.data = snail(n_points, bend, width, seed)
        else:
            self.data = snail(int(n_points/4), bend, width, seed+1)
        self.labels = torch.full((len(self.data),), 0, dtype=torch.int)
            

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.data[idx,:], self.labels[idx]

@gin.configurable('Circle')
class Circle(torch.utils.data.Dataset):
    def __init__(self, train, n_points=10000,
                 width=.05, seed = 0):
        if train == True:
            self.data = circle(n_points, width, seed)
        else:
            self.data = circle(int(n_points/4), width, seed+1)
        self.labels = torch.full((len(self.data),), 0, dtype=torch.int)
            

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.data[idx,:], self.labels[idx]

@gin.configurable('Disc')
class Disc(torch.utils.data.Dataset):
    def __init__(self, train, n_points=5000,
                 width=.05, seed = 0):
        if train == True:
            self.data = disc(n_points, width, seed)
        else:
            self.data = disc(int(n_points/4), width, seed+1)
        self.labels = torch.full((len(self.data),), 0, dtype=torch.int)
            

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.data[idx,:], self.labels[idx]
    
    
def angletocoords2d(angle, radius):
    return radius * np.array([math.cos(angle), math.sin(angle)])

def rotate2d(angle, points):
    M = np.array([[math.cos(angle), - math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    rotatedpoints = points
    for i in range(len(points)):
        rotatedpoints[i] = M.dot(points[i])
    return rotatedpoints

def flower(n_points, petals, petal_length, petal_width, seed):
    #plant the flower seed
    random.seed(seed)
    flowerpoints = np.zeros([n_points, 2])
    flowerlabels = np.zeros([n_points])
    for p in range(n_points):
        petal = random.randint(0, petals)
        flowerlabels[p] = petal
        #find center
        if petal == 0:
            center = np.array([.5, .5])
        else:
            angle0 = (petal - 1) / (petals - 1) * 2 * math.pi
            radius0 = petal_length * random.random()
            center = np.array([.5, .5]) + angletocoords2d(angle0, radius0)

        angle1 = 2 * math.pi * random.random()
        if petal == 0:
            radius1 = petal_width * random.random() / 2 #2 * petal_width * random.random()
        else:
            radius1 = petal_width * random.random() * (.75 + -abs(radius0 / petal_length - .75))
        translation = angletocoords2d(angle1, radius1)
        flowerpoints[p,:] = center + translation
    return torch.tensor(flowerpoints, dtype = torch.float), torch.tensor(flowerlabels, dtype = torch.int)

def gaussflower(n_points, petals, seed = 0):
    random.seed(seed)
    n_points = n_points - n_points % petals
    gausspoints = []
    single_num = int(n_points / petals)
    for i in range(petals):
        points = rotate2d(2 * math.pi * i / petals,  np.random.multivariate_normal([3, 0], [[1, 0], [0, 1 / (10 * petals) ]], single_num))
        gausspoints.append(points)
    gausspoints = np.vstack(gausspoints)
    return torch.tensor(gausspoints, dtype = torch.float)

def snail(n_points, bend, width, dim, seed = 0):
    random.seed(seed)
    snailpoints = np.zeros([n_points, dim])
    for k in range(n_points):
        angle = bend * 2 * math.pi * k / n_points
        radius = (n_points - k) / n_points
        angle_sway = random.random()
        radius_sway = width * random.random()
        center = np.random.uniform(-1, 1, dim)
        center[:2] = angletocoords2d(angle, radius)
        #center = angletocoords2d(angle, radius)
        translation = np.zeros(dim)
        translation[:2] = angletocoords2d(angle_sway, radius_sway)
        #translation = angletocoords2d(angle_sway, radius_sway)
        snailpoints[k, :] = (center + translation) / 2 + 0.5
    return torch.tensor(snailpoints, dtype = torch.float)

def circle(n_points, width, seed = 0):
    random.seed(seed)
    circlepoints = np.zeros([n_points, 2])
    for k in range(n_points):
        angle = 2 * math.pi * k / n_points
        radius = 1
        angle_sway = random.random() * 2 * math.pi
        radius_sway = width * random.random()
        center = angletocoords2d(angle, radius)
        translation = angletocoords2d(angle_sway, radius_sway)
        circlepoints[k, :] = (center + translation) / 2 + 0.5
    return torch.tensor(circlepoints, dtype = torch.float)

def disc(n_points, width, seed = 0):
    random.seed(seed)
    discpoints = np.zeros([n_points, 2])
    for k in range(n_points):
        if k % 2 == 0:
            angle = 2 * math.pi * k / n_points
            radius = 1
            angle_sway = random.random() * 2 * math.pi
            radius_sway = width * random.random()
            center = angletocoords2d(angle, radius)
            translation = angletocoords2d(angle_sway, radius_sway)
            discpoints[k, :] = (center + translation) / 2 + 0.5
        else:
            angle = random.random() * 2 * math.pi
            radius = math.sqrt(random.random())
            center = angletocoords2d(angle, radius)
            discpoints[k, :] = center / 2 + 0.5
    return torch.tensor(discpoints, dtype = torch.float)

#flowerpoints = flower(10000, 12, 8., 0.2, 0)
#plt.scatter(flowerpoints[:,0], flowerpoints[:,1])
#plt.savefig("flowerpoints.png")

gausspoints = gaussflower(10000, 10)
plt.scatter(gausspoints[:,0], gausspoints[:,1])
plt.savefig("gausspoints.png")
