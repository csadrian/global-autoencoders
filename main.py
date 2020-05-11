import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import os

import torch
import torch.optim

import neptune
import gin
import gin.torch
from absl import flags, app

import trainers
import models

from torchvision import datasets, transforms
import torchvision.utils as vutils

import utils

from datasets import DatasetWithIndices

@gin.configurable
class ExperimentRunner():

    def __init__(self, seed=1, no_cuda=False, num_workers=2, epochs=10, log_interval=100, plot_interval=1000, outdir='out', datadir='~/datasets', batch_size=200, prefix='', distribution='normal', dataset='mnist', ae_model_class=gin.REQUIRED):
        self.seed = seed
        self.no_cuda = no_cuda
        self.num_workers = num_workers
        self.epochs = epochs
        self.log_interval = log_interval
        self.plot_interval = plot_interval
        self.outdir = outdir
        self.datadir = datadir
        self.batch_size = batch_size
        self.prefix = prefix
        self.distribution = distribution
        self.dataset = dataset
        self.ae_model_class = ae_model_class

        self.setup_environment()
        self.setup_torch()

    def setup_environment(self):
        self.imagesdir = os.path.join(self.outdir, self.prefix, 'images')
        self.chkptdir = os.path.join(self.outdir, self.prefix, 'models')
        os.makedirs(self.datadir, exist_ok=True)
        os.makedirs(self.imagesdir, exist_ok=True)
        os.makedirs(self.chkptdir, exist_ok=True)

    def setup_torch(self):
        use_cuda = not self.no_cuda and torch.cuda.is_available()

        torch.manual_seed(self.seed)
        if use_cuda:
            torch.cuda.manual_seed(self.seed)

        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.dataloader_kwargs = {'num_workers': 3, 'pin_memory': True} if use_cuda else {'num_workers': self.num_workers, 'pin_memory': False}
        print(self.device)

    def setup_trainers(self):
        nc = 1 if self.dataset in ('mnist') else 3
        self.model = self.ae_model_class(nc=nc)
        self.model.to(self.device)
        if self.distribution == 'normal':
            #dist = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(self.model.z_dim), torcg.eye(self.model.z_dim))
            base_dist = torch.distributions.normal.Normal(torch.zeros(self.model.z_dim), torch.ones(self.model.z_dim))
            dist = torch.distributions.independent.Independent(base_dist, 1)
        elif self.distribution == 'uniform':
            base_dist = torch.distributions.uniform.Uniform(-torch.ones(self.model.z_dim), torch.ones(self.model.z_dim))
            dist = torch.distributions.independent.Independent(base_dist, 1)
        else:
            raise Exception('Distribution not implemented')
        print(dist.sample(torch.Size([5])))

        self.trainer = trainers.SinkhornTrainer(self.model, self.device, train_loader=self.train_loader, distribution=dist)

    def setup_data_loaders(self):

        if self.dataset == 'celeba':
            train_dataset = datasets.CelebA(self.datadir, split='train', target_type='attr', download=True, transform=transforms.Compose([transforms.Scale((64,64)), transforms.ToTensor()]))
            test_dataset = datasets.CelebA(self.datadir, split='test', target_type='attr', download=True, transform=transforms.Compose([transforms.Scale((64, 64)), transforms.ToTensor()]))
        elif self.dataset == 'mnist':
            train_dataset = datasets.MNIST(self.datadir, train=True, target_transform=None, download=True, transform=transforms.Compose([transforms.ToTensor()]))
            test_dataset = datasets.MNIST(self.datadir, train=False, target_transform=None, download=True, transform=transforms.Compose([transforms.ToTensor()]))
        else:
            raise Exception("Dataset not found: " + dataset)

        self.train_loader = torch.utils.data.DataLoader(DatasetWithIndices(train_dataset), batch_size=self.batch_size, shuffle=False, **self.dataloader_kwargs)
        self.test_loader = torch.utils.data.DataLoader(DatasetWithIndices(test_dataset), batch_size=self.batch_size, shuffle=False, **self.dataloader_kwargs)

    def train(self): 
        self.setup_data_loaders()
        self.setup_trainers()

        self.global_iters = 0
        for self.epoch in range(self.epochs):
            for batch_idx, (x, y, idx) in enumerate(self.train_loader, start=0):
                self.global_iters += 1
                batch = self.trainer.train_on_batch(x, idx)
                if self.global_iters % self.log_interval == 0:
                    print("Global iter: {}, Train epoch: {}, batch: {}/{}, loss: {}".format(self.global_iters, self.epoch, batch_idx+1, len(self.train_loader), batch['loss']))
                    neptune.send_metric('train_loss', x=self.global_iters, y=batch['loss'])
                    neptune.send_metric('train_reg_loss', x=self.global_iters, y=batch['reg_loss'])
                    neptune.send_metric('train_rec_loss', x=self.global_iters, y=batch['rec_loss'])

                if self.global_iters % self.plot_interval == 0:
                    self.test()

    def plot_latent_2d(self, test_encode, test_targets, test_loss):
        # save encoded samples plot
        plt.figure(figsize=(10, 10))
        plt.scatter(test_encode[:, 0], test_encode[:, 1], c=(10 * test_targets), cmap=plt.cm.Spectral)
        plt.xlim([-1.5, 1.5])
        plt.ylim([-1.5, 1.5])
        plt.title('Test Latent Space\nLoss: {:.5f}'.format(test_loss))
        filename = '{}/test_latent_epoch_{}.png'.format(self.imagesdir, self.epoch + 1)
        plt.savefig(filename)
        plt.close()
        neptune.send_image('plot_latent_2d', x=self.global_iters, y=filename)

    def plot_images(self, x, train_rec, test_rec, gen):
        utils.save_image(x, 'test_samples', self.global_iters, '{}/test_samples_epoch_{}.png'.format(self.imagesdir, self.epoch + 1))
        utils.save_image(train_rec.detach(), 'train_reconstructions', self.global_iters, '{}/train_reconstructions_epoch_{}.png'.format(self.imagesdir, self.epoch + 1), normalize=True)
        utils.save_image(test_rec.detach(), 'test_reconstructions', self.global_iters, '{}/test_reconstructions_epoch_{}.png'.format(self.imagesdir, self.epoch + 1), normalize=True)
        utils.save_image(gen.detach(), 'generated', self.global_iters, '{}/generated_epoch_{}.png'.format(self.imagesdir, self.epoch + 1), normalize=True)

    def test(self):
        test_encode, test_targets, test_loss, test_reg_loss, test_rec_loss = list(), list(), 0.0, 0.0, 0.0
        
        with torch.no_grad():
            for test_batch_idx, (x_test, y_test, idx) in enumerate(self.test_loader, start=0):
                test_evals = self.trainer.test_on_batch(x_test, idx)
                test_encode.append(test_evals['encode'].detach())
                test_loss += test_evals['loss'].item()
                test_reg_loss += test_evals['reg_loss'].item()
                test_rec_loss += test_evals['rec_loss'].item()

                test_targets.append(y_test)
        test_encode, test_targets = torch.cat(test_encode).cpu().numpy(), torch.cat(test_targets).cpu().numpy()
        test_loss /= len(self.test_loader)
        test_rec_loss /= len(self.test_loader)
        test_reg_loss /= len(self.test_loader)

        print('Test Epoch: {} ({:.2f}%)\tLoss: {:.6f}'.format(self.epoch + 1, float(self.epoch + 1) / (self.epochs) * 100., test_loss))

        neptune.send_metric('test_loss', x=self.global_iters, y=test_loss)
        neptune.send_metric('test_reg_loss', x=self.global_iters, y=test_reg_loss)
        neptune.send_metric('test_rec_loss', x=self.global_iters, y=test_rec_loss)
        
        self.plot_latent_2d(test_encode, test_targets, test_loss)
    
        test_batch, (x, y, idx) = enumerate(self.test_loader, start=0).__next__()
        test_batch = self.trainer.test_on_batch(x, idx)

        train_batch, (x, y, idx) = enumerate(self.train_loader, start=0).__next__()
        train_batch = self.trainer.test_on_batch(x, idx)
        gen_batch = self.trainer.decode_batch(self.trainer.sample_pz(n=self.batch_size))
        self.plot_images(x, train_batch['decode'], test_batch['decode'], gen_batch['decode'])


def main(argv):
    gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_param, skip_unknown=True)
    print(gin.operative_config_str())
    
    use_neptune = "NEPTUNE_API_TOKEN" in os.environ
    if use_neptune:
        neptune.init(project_qualified_name="csadrian/global-autoencoders")
        exp = neptune.create_experiment(params={}, name="exp")
        #for tag in opts['tags'].split(','):
        #  neptune.append_tag(tag)
    else:
        neptune.init('shared/onboarding', api_token='ANONYMOUS', backend=neptune.OfflineBackend())

    er = ExperimentRunner(prefix=exp.id)
    er.train()

    neptune.stop()
    print('fin')

if __name__ == '__main__':
    flags.DEFINE_multi_string('gin_file', None, 'List of paths to the config files.')
    flags.DEFINE_multi_string('gin_param', None, 'Newline separated list of Gin parameter bindings.')
    FLAGS = flags.FLAGS

    app.run(main)