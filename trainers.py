  
import numpy as np

import torch
import torch.nn.functional as F

from torch.autograd import Variable

from geomloss import SamplesLoss

import gin
import gin.torch


@gin.configurable
class SinkhornTrainer:
    def __init__(self, model, device, optimizer=torch.optim.Adam, distribution_class=torch.distributions.multivariate_normal.MultivariateNormal, reg_lambda=1.0): #gin.REQUIRED):
        self.model = model
        self.device = device
        self.distribution = distribution_class(torch.zeros(self.model.z_dim), torch.eye(self.model.z_dim))
        #self.z_dim = self.autoencoder.z_dim
        self.reg_lambda = reg_lambda
        self.optimizer = optimizer(self.model.parameters())

    def sample_pz(self, n=100):
        return self.distribution.sample(torch.Size([n]))

    def decode_batch(self, z):
        z = z.to(self.device)
        gen_x = self.model._decode(z)
        return {
            'decode': gen_x
        }

    def test_on_batch(self, x):
        x = x.to(self.device)
        recon_x, z = self.model(x)
        # mutual information reconstruction loss
        bce = F.mse_loss(recon_x, x)

        # divergence on transformation plane from X space to Z space to match prior
        #_swd = sliced_wasserstein_distance(z, self._distribution_fn,
        #                                   self.num_projections_, self.p_,
        #                                   self._device)
        #w2 = float(self.weight) * _swd  # approximate wasserstein-2 distance

        reg_loss_fn = SamplesLoss(loss="sinkhorn", p=2, blur=.05)
        pz_sample = self.sample_pz(z.size()[0]).to(self.device)
        reg_loss = reg_loss_fn(z, pz_sample)  # By default, use constant weights = 1/number of samples


        loss = bce + float(self.reg_lambda) * reg_loss

        return {
            'loss': loss,
            #'bce': bce,
            #'l1': l1,
            #'w2': w2,
            'encode': z,
            'decode': recon_x
        }
        pass        

    def train_on_batch(self, x):
        self.optimizer.zero_grad()
        result = self.test_on_batch(x)
        result['loss'].backward()
        self.optimizer.step()
        return result

