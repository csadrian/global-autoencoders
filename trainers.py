import numpy as np

import gc

import torch
import torch.nn.functional as F

from torch.autograd import Variable

from geomloss import SamplesLoss

import gin
import gin.torch
import gin.torch.external_configurables

import synthetic

import time

@gin.configurable
class SinkhornTrainer:
    def __init__(self, model, device, batch_size, optimizer=gin.REQUIRED, distribution=gin.REQUIRED, reg_lambda=gin.REQUIRED, nat_size=None, 
                    train_loader=None, test_loader=None, trainer_type='global', monitoring = True, sinkhorn_scaling = 0.5, resampling_freq = 1, recalculate_freq = 1, 
                    reg_loss_type = 'sinkhorn', blur = None, trail_label_idx=0,
                 full_backprop = False, rec_cost = 'mean', simplex_var = 0.05, sphererad = 0.5):
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
        assert trainer_type in {'local', 'global'}, "trainer_type has to be `local` or `global`"
        self.trainer_type = trainer_type
        self.resampling_freq = resampling_freq
        self.recalculate_freq = recalculate_freq
        self.trail_label_idx = trail_label_idx
        self.full_backprop = full_backprop

        assert rec_cost in {'mean', 'sum', 'Tolstikhin', 'Tolstikhin2'}
        self.rec_cost = rec_cost

        self.simplex_var = simplex_var
        self.sphererad = sphererad
        
        #In the local, no monitoring case, generate video and covered area from fixed batch.
        _, (self.trail_batch, self.trail_labels, _) = enumerate(self.train_loader).__next__()
        self.trail_batch = self.trail_batch.to(self.device)
        self.trail_labels = self.trail_labels
        label_type = self.trail_labels.dtype
        self.trail_labels = self.trail_labels.cpu().detach().numpy()
        
        self.all_labels = torch.zeros(torch.Size([len(train_loader.dataset)]), dtype=label_type)
        for _, (_, y, idx) in enumerate(self.train_loader):
            with torch.no_grad():
                self.all_labels[idx] = y if len(self.trail_labels.shape) < 2 else y[:, self.trail_label_idx]
        self.all_labels = self.all_labels.cpu().detach().numpy()

        # If there are multiple labels, use trail_label_idx
        if len(self.trail_labels.shape) == 2:
            self.trail_labels = self.trail_labels[self.trail_label_idx]

 
        if reg_loss_type in {'sinkhorn', 'gaussian', 'energy', 'laplacian', 'IMQ'}:
            cost = None
            if blur:
                self.blur = blur
            #else set default blur parameter using heuristics
            elif reg_loss_type == 'sinkhorn':
                self.blur = 0.05
            elif reg_loss_type == 'gaussian':
                #this is sigma_median_heuristic value for mnist, tune otherwise
                self.blur = 4.
            elif reg_loss_type == 'IMQ':
                cost = "IntCst(1)/(IntCst(1)+SqDist(X,Y))"
                #specifies kernel method, kernel determined by cost
                reg_loss_type = 'gaussian'
                self.blur = np.sqrt(2*self.model.z_dim)
            elif reg_loss_type == 'laplacian':
                self.blur = 1.
                
            self.reg_loss_fn = SamplesLoss(loss=reg_loss_type, cost = cost, p=2, blur=self.blur, backend='online', scaling = self.sinkhorn_scaling, verbose=True)
            self.cost = cost
            self.reg_loss_type = reg_loss_type
        else:
            assert False, 'reg_loss not implemented'
            
        #If nat_size unspecified, initialize.
        if nat_size is None:
            if trainer_type == 'global':
                nat_size = len(train_loader.dataset)
            elif trainer_type == 'local':
                nat_size = self.batch_size

        self.nat_size = nat_size
        self.pz_sample = self.sample_pz(self.nat_size).to(self.device)

        # Save all x_latents for global_train_reg_loss.
        self.x_latents = torch.zeros(torch.Size([len(train_loader.dataset), self.model.z_dim])).to(self.device).detach()

    def sample_pz(self, n=100):
        if self.distribution in {'normal', 'sphere'}:
            base_dist = torch.distributions.normal.Normal(torch.zeros(self.model.z_dim), torch.ones(self.model.z_dim))
            dist = torch.distributions.independent.Independent(base_dist, 1)
        elif self.distribution == 'flower':
            seed = np.random.randint(0, 10000)
            #ONLY 2-dim
            sample, _ = synthetic.flower(n, 10, .5, .2, seed)#30, .5, .03, seed)
            return sample
        elif self.distribution == 'snail':
            seed = np.random.randint(0, 10000)
            #ONLY 2-dim
            sample = synthetic.snail(n, 2.5, 1, 2, seed)
            return sample
        elif self.distribution == 'gaussflower':
            seed = np.random.randint(0, 10000)
            #ONLY 2-dim
            sample = synthetic.gaussflower(n, 10, seed)
            return sample
        elif self.distribution == 'gaussimplex':
            seed = np.random.randint(0, 10000)
            #ONLY 2-dim
            sample = synthetic.gaussimplex(n, 10, self.simplex_var, seed)
            return sample
        elif self.distribution == 'simplexflower':
            seed = np.random.randint(0, 10000)
            #ONLY 2-dim
            sample = synthetic.simplexflower(n, 10, self.simplex_var, seed)
            return sample
        elif self.distribution == 'simplexsphere':
            seed = np.random.randint(0, 10000)
            #ONLY 2-dim
            sample = synthetic.spheresimplex(n, 10, self.sphererad, seed)
            return sample
        elif self.distribution == 'uniform':
            base_dist = torch.distributions.uniform.Uniform(-torch.ones(self.model.z_dim), torch.ones(self.model.z_dim))
            dist = torch.distributions.independent.Independent(base_dist, 1)
        else:
            raise Exception('Distribution not implemented')
        #print(dist.sample(torch.Size([5])))

        if self.distribution == 'sphere':
            s = dist.sample(torch.Size([n]))
            norm = torch.norm(s, p=2, dim=1)[:, np.newaxis]
            return s / norm
        else:
            sample = dist.sample(torch.Size([n]))
            return sample


    def decode_batch(self, z):
        with torch.no_grad():
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
            recon_x, _ = self.model(x)
            del x
            return recon_x
        
    def rec_loss_on_test(self, x_test):
        with torch.no_grad():
            x = x_test.to(self.device)
            recon_x, z = self.model(x)
            if self.rec_cost == 'Tolstikhin':
                rec_loss = 0.05 * F.mse_loss(recon_x,x, reduction='sum') / 100.
            elif self.rec_cost == 'Tolstikhin2':
                rec_loss = 0.05 * F.mse_loss(recon_x,x)
            else:
                rec_loss = F.mse_loss(recon_x,x, reduction=self.rec_cost)
            del x
            return {
                'encode': z,
                'rec_loss': rec_loss
                }
        

    def sigma_median_heuristic(self, X, Y):
        with torch.no_grad():
            p2_norm_x = X.pow(2).sum(1).unsqueeze(0)
            prods_x = torch.mm(X, X.t())
            dists_x = p2_norm_x + p2_norm_x.t() - 2 * prods_x

            p2_norm_y = Y.pow(2).sum(1).unsqueeze(0)
            prods_y = torch.mm(Y, Y.t())
            dists_y = p2_norm_y + p2_norm_y.t() - 2 * prods_y

            dot_prd = torch.mm(X, Y.t())
            dists_c = p2_norm_x + p2_norm_y.t() - 2 * dot_prd

            sigma2_k = torch.median(torch.reshape(dists_c, (-1,)))
            sigma2_k += torch.median(torch.reshape(dists_y, (-1,)))
            return sigma2_k
            
    def loss_on_batch(self, x, curr_indices, iter):
        recalc_latents = not iter % self.recalculate_freq
        resample = not iter % self.resampling_freq

        x = x.to(self.device)
        recon_x, z = self.model(x)
        
        #recompute latent points
        if self.trainer_type == 'global' or self.monitoring:
            #detach from any previous computations
            self.x_latents = self.x_latents.detach()
            if recalc_latents:
                self.recalculate_latents()
            #if not(recalc_latents and self.full_backprop):
            self.x_latents.index_copy_(0, curr_indices.to(self.device), z)
            video_batch = self.x_latents
            video_labels = self.all_labels
        else:
            with torch.no_grad():
                video_batch = self.model._encode(self.trail_batch)
                video_labels = self.trail_labels

        if self.rec_cost == 'Tolstikhin':
            bce = 0.05 * F.mse_loss(recon_x, x, reduction='sum') / 100.
        elif self.rec_cost == 'Tolstikhin2':
            bce = 0.05 * F.mse_loss(recon_x, x)    
        else:
            bce = F.mse_loss(recon_x, x, reduction=self.rec_cost)
        
        if self.reg_lambda != 0.0:
            if resample:
                self.pz_sample = self.sample_pz(self.nat_size).to(self.device)

            if self.trainer_type == 'local':
                z_prime = z
            elif self.trainer_type == 'global':
                z_prime = self.x_latents

            reg_loss = self.reg_loss_fn(z_prime, self.pz_sample.detach())  # By default, use constant weights = 1/number of samples
            loss = bce + float(self.reg_lambda) * reg_loss
            #loss = float(self.reg_lambda) * reg_loss
        else:
            z_prime = z
            loss = bce.clone()
            reg_loss = 0.0

        result = {
            'loss': loss,
            'reg_loss': reg_loss,
            'rec_loss': bce,
            'reg_lambda': self.reg_lambda,
            'encode': z,
            'decode': recon_x,
            'video': {'latents': video_batch, 'labels': video_labels},
            'blur' : self.blur
        }
        return result

    def train_on_batch(self, x, curr_indices, iter):

        self.optimizer.zero_grad()
        '''
        totalmem = 0
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                    mem = obj.nelement() * obj.element_size() / 1024 / 1024
                    totalmem += mem
                    print(type(obj), obj.size(), mem)
            except:
                pass
        print(totalmem)
        '''
        result = self.loss_on_batch(x, curr_indices, iter)
        result['loss'].backward()
        self.optimizer.step()
        return result

    def recalculate_latents(self):
        if self.reg_lambda == 0.0:
            return

        if self.full_backprop:
            for batch_idx, (x, y, idx) in enumerate(self.train_loader):
                x = x.to(self.device)
                self.x_latents[idx] = self.model._encode(x)
                del x
        else:
            for batch_idx, (x, y, idx) in enumerate(self.train_loader):
                with torch.no_grad():
                    x = x.to(self.device)
                    self.x_latents[idx] = self.model._encode(x)
                    del x
