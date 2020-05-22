
import numpy as np

import torch
import torch.nn.functional as F

from torch.autograd import Variable

from geomloss import SamplesLoss

import gin
import gin.torch


@gin.configurable
class SinkhornTrainer:
    def __init__(self, model, device, optimizer=torch.optim.Adam, distribution=gin.REQUIRED, reg_lambda=1.0, nat_size=None, batch_size=64, train_loader=None, test_loader=None, type='global', monitoring = True, sinkhorn_scaling = 0.3, resampling_freq = 1, recalculate_freq = 1):
        self.model = model
        self.device = device
        self.distribution = distribution
        #self.z_dim = self.autoencoder.z_dim
        self.reg_lambda = reg_lambda
        self.batch_size = batch_size
        self.optimizer = optimizer(self.model.parameters())
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.monitoring = monitoring
        self.sinkhorn_scaling = sinkhorn_scaling
        assert type in {'local', 'global'}, "type has to be `local` or `global`"
        self.type = type
        self.resampling_freq = resampling_freq
        self.recalculate_freq = recalculate_freq

        self.reg_loss_fn = SamplesLoss(loss="sinkhorn", p=2, blur=.05, backend='online', scaling = self.sinkhorn_scaling, verbose=True)
        #If nat_size unspecified, initialize.
        if nat_size is None:
            if type == 'global':
                nat_size = len(train_loader.dataset)
            elif type == 'local':
                nat_size = batch_size

        self.nat_size = nat_size
        self.pz_sample = self.sample_pz(self.nat_size).to(self.device)

        # If monitoring, save all x_latents. Otherwise only nat_size many x_latents are required.
        if monitoring:
            self.x_latents = torch.zeros(torch.Size([len(train_loader.dataset), self.model.z_dim])).to(self.device).detach()
        else:
            self.x_latents = torch.zeros(torch.Size([self.nat_size, self.model.z_dim])).to(self.device).detach()

    def sample_pz(self, n=100):
        if self.distribution == 'normal' or 'sphere':
            base_dist = torch.distributions.normal.Normal(torch.zeros(self.model.z_dim), torch.ones(self.model.z_dim))
            dist = torch.distributions.independent.Independent(base_dist, 1)
        elif self.distribution == 'uniform':
            base_dist = torch.distributions.uniform.Uniform(-torch.ones(self.model.z_dim), torch.ones(self.model.z_dim))
            dist = torch.distributions.independent.Independent(base_dist, 1)
        else:
            raise Exception('Distribution not implemented')
        #print(dist.sample(torch.Size([5])))

        if self.distribution == 'sphere':
            s = dist.sample(torch.Size([n]))
            n = torch.norm(s, p=2, dim=1)[:, np.newaxis]
            return s / n
        else:
            return dist.sample(torch.Size([n]))


    def decode_batch(self, z):
        z = z.to(self.device)
        gen_x = self.model._decode(z)
        return {
            'decode': gen_x
        }

    def reg_loss_on_test(self):
        with torch.no_grad():
            test_latents = torch.zeros(torch.Size([len(self.test_loader.dataset), self.model.z_dim])).to(self.device).detach()
            for batch_idx, (x, y, idx) in enumerate(self.test_loader):
                x = x.to(self.device)
                test_latents[idx] = self.model._encode(x)
                del x
            pz_sample = self.sample_pz(len(self.test_loader.dataset)).to(self.device).detach()
            reg_loss = self.reg_loss_fn(test_latents, pz_sample.detach())
            del pz_sample
            del test_latents
            return reg_loss
            
    def reconstruct(self, x):
        with torch.no_grad():
            x = x.to(self.device)
            recon_x, z = self.model(x)
            del x
            del z
            return recon_x
        
    def rec_loss_on_test(self, x_test):
        with torch.no_grad():
            x = x_test.to(self.device)
            recon_x, z = self.model(x)
            rec_loss = F.mse_loss(recon_x,x)
            del x
            return {
                'encode': z,
                'rec_loss': rec_loss
                }
        
        
    def loss_on_batch(self, x, curr_indices, iter):

        recalc_latents = not iter % self.recalculate_freq
        resample = not iter % self.resampling_freq
        
        x = x.to(self.device)
        recon_x, z = self.model(x)
        
        #recompute latent points
        if self.type == 'global' or self.monitoring:
            #detach from any previous computations
            self.x_latents = self.x_latents.detach()
            if recalc_latents:
                self.recalculate_latents()
            self.x_latents.index_copy_(0, curr_indices.to(self.device), z)
        
        bce = F.mse_loss(recon_x, x)
        
        if self.reg_lambda != 0.0:
            if resample:
                self.pz_sample = self.sample_pz(self.nat_size).to(self.device)

            if self.type == 'local':
                z_prime = z
            elif self.type == 'global':
                z_prime = self.x_latents
            reg_loss = self.reg_loss_fn(z_prime, self.pz_sample.detach())  # By default, use constant weights = 1/number of samples
            loss = bce + float(self.reg_lambda) * reg_loss
        else:
            loss = bce.clone()
            reg_loss = 0.0

        return {
            'loss': loss,
            'reg_loss': reg_loss,
            'rec_loss': bce,
            'reg_lambda': self.reg_lambda,
            'encode': z,
            'decode': recon_x,
            'full_encode': z_prime
        }

    def train_on_batch(self, x, curr_indices, iter):

        self.optimizer.zero_grad()

        result = self.loss_on_batch(x, curr_indices, iter)
        result['loss'].backward()
        self.optimizer.step()
        return result

    def recalculate_latents(self):
        if self.reg_lambda == 0.0:
            return

        for batch_idx, (x, y, idx) in enumerate(self.train_loader):
            with torch.no_grad():
                x = x.to(self.device)
                self.x_latents[idx] = self.model._encode(x)
                del x
 
