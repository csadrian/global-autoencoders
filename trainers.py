
import numpy as np

import torch
import torch.nn.functional as F

from torch.autograd import Variable

from geomloss import SamplesLoss

import gin
import gin.torch


@gin.configurable
class SinkhornTrainer:
    def __init__(self, model, device, optimizer=torch.optim.Adam, distribution=gin.REQUIRED, reg_lambda=1.0, nat_size=None, batch_size=64, train_loader=None):
        self.model = model
        self.device = device
        self.distribution = distribution
        #self.z_dim = self.autoencoder.z_dim
        self.reg_lambda = reg_lambda
        self.batch_size = batch_size
        self.optimizer = optimizer(self.model.parameters())
        self.train_loader = train_loader

        if nat_size is None:
            nat_size = len(train_loader)
        self.nat_size = nat_size


    def sample_pz(self, n=100):
        return self.distribution.sample(torch.Size([n]))

    def decode_batch(self, z):
        z = z.to(self.device)
        gen_x = self.model._decode(z)
        return {
            'decode': gen_x
        }

    def test_on_batch(self, x, curr_indices):
        x = x.to(self.device)
        recon_x, z = self.model(x)

        bce = F.mse_loss(recon_x, x)

        self.recalculate_latents()

        reg_loss_fn = SamplesLoss(loss="sinkhorn", p=2, blur=.05, backend='online', scaling=0.3, verbose=True)
        #reg_loss_fn = SamplesLoss(loss="gaussian", p=2, blur=.05, backend='online', scaling=0.01, verbose=True)
        pz_sample = self.sample_pz(self.nat_size).to(self.device)
        
        print(curr_indices)
        z_prime = self.x_latents[curr_indices] = z

        if self.reg_lambda != 0.0:
            reg_loss = reg_loss_fn(z_prime, pz_sample.detach())  # By default, use constant weights = 1/number of samples
        else:
            reg_loss = 0.0

        loss = bce + float(self.reg_lambda) * reg_loss

        return {
            'loss': loss,
            'reg_loss': reg_loss,
            'encode': z,
            'decode': recon_x
        }

    def train_on_batch(self, x, curr_indices):

        self.optimizer.zero_grad()

        result = self.test_on_batch(x, curr_indices)
        result['loss'].backward()
        self.optimizer.step()
        return result

    def recalculate_latents(self):

        x_latents_a = list()
        it = 0
        for batch_idx, (x, y, idx) in enumerate(self.train_loader):
          it += 1
          with torch.no_grad():
            x = x.to(self.device)
            x_latents_a.append(self.model._encode(x))
            del x
            #print(it, len(self.train_loader))
        self.x_latents = torch.cat(x_latents_a)
