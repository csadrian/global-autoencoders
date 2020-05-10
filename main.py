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

from datasets import DatasetWithIndices

@gin.configurable
class ExperimentRunner():

    def __init__(self, seed=1, no_cuda=False, num_workers=2, epochs=10, log_interval=100, outdir='out', datadir='~/datasets', batch_size=200, prefix=''):
        self.seed = seed
        self.no_cuda = no_cuda
        self.num_workers = num_workers
        self.epochs = epochs
        self.log_interval = log_interval
        self.outdir = outdir
        self.datadir = datadir
        self.batch_size = batch_size
        self.prefix = prefix

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
    
    def setup_trainers(self):
        self.model = models.WAE()
        self.model.to(self.device)
        self.trainer = trainers.SinkhornTrainer(self.model, self.device, train_loader=self.train_loader)

    def setup_data_loaders(self):

        self.train_loader = torch.utils.data.DataLoader(DatasetWithIndices(
            datasets.CelebA(self.datadir, split='train', target_type='attr', download=True,
                transform=transforms.Compose([transforms.Scale((64,64)), transforms.ToTensor()])#, transforms.Normalize((0.1307,), (0.3081,))])
            )),
            batch_size=self.batch_size, shuffle=False, **self.dataloader_kwargs)
 
        self.test_loader = torch.utils.data.DataLoader(DatasetWithIndices(
            datasets.CelebA(self.datadir, split='test', target_type='attr', download=True,
                transform=transforms.Compose([transforms.Scale((64, 64)), transforms.ToTensor()])#, transforms.Normalize((0.1307,), (0.3081,))])
            )),
            batch_size=self.batch_size, shuffle=False, **self.dataloader_kwargs)

    def train(self): 
        self.setup_data_loaders()
        self.setup_trainers()

        iteration = 0
        for self.epoch in range(self.epochs):
            for batch_idx, (x, y, idx) in enumerate(self.train_loader, start=0):
                iteration += 1

                batch = self.trainer.train_on_batch(x, idx)
                if (batch_idx + 1) % self.log_interval == 0:
                    print("Global iter: {}, Train epoch: {}, batch: {}/{}, loss: {}".format(iteration, self.epoch, batch_idx+1, len(self.train_loader), batch['loss']))
                    self.test()

    def plot_latent_mnist(self, test_encode, test_targets, test_loss):
        # save encoded samples plot
        plt.figure(figsize=(10, 10))
        plt.scatter(test_encode[:, 0], -test_encode[:, 1], c=(10 * test_targets), cmap=plt.cm.Spectral)
        plt.xlim([-1.5, 1.5])
        plt.ylim([-1.5, 1.5])
        plt.title('Test Latent Space\nLoss: {:.5f}'.format(test_loss))
        plt.savefig('{}/test_latent_epoch_{}.png'.format(self.imagesdir, epoch + 1))
        plt.close()

    def plot_images(self, x, train_rec, test_rec, gen):
        vutils.save_image(x, '{}/test_samples_epoch_{}.png'.format(self.imagesdir, self.epoch + 1))
        vutils.save_image(train_rec.detach(), '{}/train_reconstructions_epoch_{}.png'.format(self.imagesdir, self.epoch + 1), normalize=True)
        vutils.save_image(test_rec.detach(), '{}/test_reconstructions_epoch_{}.png'.format(self.imagesdir, self.epoch + 1), normalize=True)
        vutils.save_image(gen.detach(), '{}/generated_epoch_{}.png'.format(self.imagesdir, self.epoch + 1), normalize=True)

    def test(self):
        test_encode, test_targets, test_loss = list(), list(), 0.0
        
        """
        with torch.no_grad():
            for test_batch_idx, (x_test, y_test, idx) in enumerate(self.test_loader, start=0):
                test_evals = self.trainer.test_on_batch(x_test)
                test_encode.append(test_evals['encode'].detach())
                test_loss += test_evals['loss'].item()
                test_targets.append(y_test)
        test_encode, test_targets = torch.cat(test_encode).cpu().numpy(), torch.cat(test_targets).cpu().numpy()
        test_loss /= len(self.test_loader)
        print('Test Epoch: {} ({:.2f}%)\tLoss: {:.6f}'.format(
                self.epoch + 1, float(self.epoch + 1) / (self.epochs) * 100.,
                test_loss))
        print('{{"metric": "loss", "value": {}}}'.format(test_loss))
        """ 
        #self.plot_latent(test_encode, test_targets, test_loss)
    
        test_batch, (x, y) = enumerate(self.test_loader, start=0).__next__()
        test_batch = self.trainer.test_on_batch(x)

        train_batch, (x, y) = enumerate(self.train_loader, start=0).__next__()
        train_batch = self.trainer.test_on_batch(x)
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