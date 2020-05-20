
import numpy as np

import torch
import torch.nn.functional as F

from torch.autograd import Variable

from geomloss import SamplesLoss

import gin
import gin.torch


@gin.configurable
class SinkhornTrainer:
    def __init__(self, model, device, optimizer=torch.optim.Adam, distribution=gin.REQUIRED, reg_lambda=1.0, nat_size=None, batch_size=64, train_loader=None, type='global', monitoring = True):
        self.model = model
        self.device = device
        self.distribution = distribution
        #self.z_dim = self.autoencoder.z_dim
        self.reg_lambda = reg_lambda
        self.batch_size = batch_size
        self.optimizer = optimizer(self.model.parameters())
        self.train_loader = train_loader
        self.monitoring = monitoring

        assert type in {'local', 'global'}, "type has to be `local` or `global`"
        self.type = type

        # If nat_size unspecified, initialize.
        if nat_size is None:
            if type == 'global':
                nat_size = len(train_loader.dataset)
            elif type == 'local':
                nat_size = batch_size

        self.nat_size = nat_size

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

    def test_on_batch(self, x, curr_indices, batch_index = None):        
        x = x.to(self.device)
        recon_x, z = self.model(x)

        if self.type == 'global' or self.monitoring:
            self.recalculate_latents(batch_index)
            self.x_latents = torch.cat((self.x_latents, z), dim=0)
        
        bce = F.mse_loss(recon_x, x)

        reg_loss_fn = SamplesLoss(loss="sinkhorn", p=2, blur=.05, backend='online', scaling=0.3, verbose=True)
        #reg_loss_fn = SamplesLoss(loss="gaussian", p=2, blur=.05, backend='online', scaling=0.01, verbose=True)
                
        loss = bce

        if self.reg_lambda != 0.0:
            pz_sample = self.sample_pz(self.nat_size).to(self.device)

            if self.type == 'local':
                z_prime = z
            elif self.type == 'global':
                z_prime = self.x_latents

            reg_loss = reg_loss_fn(z_prime, pz_sample.detach())  # By default, use constant weights = 1/number of samples
            
            loss += float(self.reg_lambda) * reg_loss
        else:
            reg_loss = 0.0

        return {
            'loss': loss,
            'reg_loss': reg_loss,
            'rec_loss': bce,
            'reg_lambda': self.reg_lambda,
            'encode': z,
            'decode': recon_x
        }

    def train_on_batch(self, x, curr_indices, batch_index):

        self.optimizer.zero_grad()

        result = self.test_on_batch(x, curr_indices, batch_index)
        result['loss'].backward()
        self.optimizer.step()
        return result

    def recalculate_latents(self, batch_exclude = None):
        if self.reg_lambda == 0.0:
            return

        #RECALCULATES ALL LATENTS EXCEPT FOR BATCH batch_exclude
        x_latents_a = list()
        it = 0
        for batch_idx, (x, y, idx) in enumerate(self.train_loader):
            it += 1
            with torch.no_grad():
                if batch_idx != batch_exclude:
                    x = x.to(self.device)
                    x_latents_a.append(self.model._encode(x))
                    #idx = idx.to(self.device)
                    #self.x_latents.index_copy_(0, idx, self.model._encode(x))
                    del x
        self.x_latents = torch.cat(x_latents_a)
            #print(it, len(self.train_loader))
