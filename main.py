import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.neighbors import NearestNeighbors

import os, sys
import gc

import torch
import torch.optim

import PIL

import neptune
import gin
import gin.torch
from absl import flags, app

import trainers
import models
import visual

from torchvision import datasets, transforms
import torchvision.utils as vutils

import utils

import synthetic

from datasets import DatasetWithIndices

from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter
from scipy.stats import norm
from scipy.stats import multivariate_normal
import math

import networkx as nx
from networkx.algorithms import bipartite
from networkx.algorithms.matching import max_weight_matching

@gin.configurable('ExperimentRunner')
class ExperimentRunner():


    def __init__(self, seed=1, no_cuda=False, num_workers=2, epochs=None, log_interval=100, plot_interval=1000, outdir='out', datadir='~/datasets', batch_size=200, num_iterations= None,  prefix='', dataset='mnist', ae_model_class=gin.REQUIRED, resampling_freq = 1, recalculate_freq = 1, limit_train_size=None, trail_label_idx=0, full_video = False, input_normalize_sym = False):
        self.seed = seed
        self.no_cuda = no_cuda
        self.num_workers = num_workers
        if epochs and num_iterations:
            assert False, 'Please only specify either epochs or iterations.'
        elif epochs == None and num_iterations == None:
            assert False, 'Please specify epochs or iterations.'
        else:
            self.epochs = epochs
            self.num_iterations = num_iterations
        self.log_interval = log_interval
        self.plot_interval = plot_interval
        self.outdir = outdir
        self.datadir = datadir
        self.batch_size = batch_size
        self.prefix = prefix
        self.dataset = dataset
        self.ae_model_class = ae_model_class
        self.limit_train_size = limit_train_size
        self.trail_label_idx = trail_label_idx
        self.full_video = full_video
        self.input_normalize_sym = input_normalize_sym
        
        self.setup_environment()
        self.setup_torch()

    def setup_environment(self):
        self.imagesdir = os.path.join(self.outdir, self.prefix, 'images')
        self.chkptdir = os.path.join(self.outdir, self.prefix, 'models')
        self.viddir = os.path.join(self.outdir, self.prefix, 'videos')
        os.makedirs(self.datadir, exist_ok=True)
        os.makedirs(self.imagesdir, exist_ok=True)
        os.makedirs(self.chkptdir, exist_ok=True)
        os.makedirs(self.viddir, exist_ok=True)

    def setup_torch(self):
        use_cuda = not self.no_cuda and torch.cuda.is_available()

        torch.manual_seed(self.seed)
        if use_cuda:
            torch.cuda.manual_seed(self.seed)

        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.dataloader_kwargs = {'num_workers': 3, 'pin_memory': True} if use_cuda else {'num_workers': self.num_workers, 'pin_memory': False}
        print(self.device)

    def setup_trainers(self):
        if self.dataset in ('mnist', 'fashionmnist', 'kmnist'):
            input_dims = (28, 28, 1)
            nc = 1
        elif self.dataset in ('flower', 'snail', 'circle', 'disc'):
            input_dims = (2,)
            nc = 1
        elif self.dataset in ('square'):
            input_dims = (2,)
            nc = 1
        else:
            input_dims = (64, 64, 3)
            nc = 3

        self.model = self.ae_model_class(nc=nc, input_dims=input_dims, input_normalize_sym=self.input_normalize_sym)
        self.model.to(self.device)
        self.distribution = self.model.distribution

        self.trainer = trainers.SinkhornTrainer(self.model, self.device, batch_size = self.batch_size, train_loader=self.train_loader, test_loader=self.test_loader, distribution=self.distribution, trail_label_idx = self.trail_label_idx)

    def normalize_latents(self, z):
        latents = z
        latents = latents.cpu()
        latents = latents.detach().numpy()
        if self.distribution == 'normal':
            latents = norm.cdf(latents) * 2 - 1
        if self.distribution == 'sphere':    
            latents = norm.cdf(latents * math.sqrt(self.model.z_dim)) * 2 - 1
        return latents[:, :2]

    def split_to_petals(self, z):
        latents = z
        latents = latents.cpu()
        latents = latents.detach().numpy()
        points_as_petals = [[] for i in range(10)]
        indices = [[] for i in range(10)]
        for i in range(len(latents)):
            c = complex(latents[i][0], latents[i][1])
            angle = np.angle(c)
            if angle < 0:
                angle += 2 * math.pi
            idx = int((int(angle * 10 / math.pi) + 1) / 2) % 10
            points_as_petals[idx].append(latents[i])
            indices[idx].append(i)
        return points_as_petals, indices

    def split_to_vertex(self, z):
        latents = z
        latents = latents.cpu()
        latents = latents.detach().numpy()
        points_as_vertex = [[] for i in range(10)]
        indices = [[] for i in range(10)]
        M = np.identity(10)
        for i in range(len(latents)):
            v = latents[i] - M
            idx = np.argmin(np.linalg.norm(v, axis = 1))
            points_as_vertex[idx].append(latents[i])
            indices[idx].append(i)
        return points_as_vertex, indices
    
    def gaussflower_covered(self, z):
        points_as_petals, _ = self.split_to_petals(z)
        av = 0
        for j in range(10):
            points = np.asarray(points_as_petals[j])
            points = synthetic.rotate2d(- (2 * math.pi / 10) * j, points)
            #points = multivariate_normal.cdf(points, mean = [3, 0], cov = [[1, 0], [0, 1 / 100]])
            M = np.array([[1, 0], [0, 10]])
            v = np.array([3, 0])
            points = np.matmul(points, M) - v
            points = norm.cdf(points) * 2 - 1
            av += visual.covered_area(points)
        return av / 10

    def gaussimplex_covered(self, z):
        points_as_vertex, _ = self.split_to_vertex(z)
        av = 0
        for j in range(10):
            points = np.asarray(points_as_vertex[j])
            M = np.identity(10) * math.sqrt(20)
            v = np.zeros(10)
            v[j] = math.sqrt(20)
            points = np.matmul(points, M) - v
            points = norm.cdf(points) * 2 - 1
            av += visual.covered_area(points[:,:2])
        return av / 10

    def setup_data_loaders(self):

        if self.dataset == 'celeba':
            transform_list = [transforms.CenterCrop(140), transforms.Resize((64,64),PIL.Image.ANTIALIAS), transforms.ToTensor()]
            #if self.input_normalize_sym:
                #D = 64*64*3
                #transform_list.append(transforms.LinearTransformation(2*torch.eye(D), -.5*torch.ones(D)))
            transform = transforms.Compose(transform_list)
            train_dataset = datasets.CelebA(self.datadir, split='train', target_type='attr', download=True, transform=transform)
            test_dataset = datasets.CelebA(self.datadir, split='test', target_type='attr', download=True, transform=transform)
            self.nlabels = 0
        elif self.dataset == 'mnist':
            train_dataset = datasets.MNIST(self.datadir, train=True, target_transform=None, download=True, transform=transforms.Compose([transforms.ToTensor()]))
            test_dataset = datasets.MNIST(self.datadir, train=False, target_transform=None, download=True, transform=transforms.Compose([transforms.ToTensor()]))
            self.nlabels = 10
        elif self.dataset == 'fashionmnist':
            train_dataset = datasets.FashionMNIST(self.datadir, train=True, target_transform=None, download=True, transform=transforms.Compose([transforms.ToTensor()]))
            test_dataset = datasets.FashionMNIST(self.datadir, train=False, target_transform=None, download=True, transform=transforms.Compose([transforms.ToTensor()]))
            self.nlabels = 10
        elif self.dataset == 'kmnist':
            train_dataset = datasets.KMNIST(self.datadir, train=True, target_transform=None, download=True, transform=transforms.Compose([transforms.ToTensor()]))
            test_dataset = datasets.KMNIST(self.datadir, train=False, target_transform=None, download=True, transform=transforms.Compose([transforms.ToTensor()]))
            self.nlabels = 10
        elif self.dataset == 'flower':
            train_dataset = synthetic.Flower(train = True)
            test_dataset = synthetic.Flower(train = False)
            self.nlabels = train_dataset.petals
        elif self.dataset == 'square':
            train_dataset = synthetic.SquareGrid(train = True)
            test_dataset = synthetic.SquareGrid(train = False)
            self.nlabels = 0            
        elif self.dataset == 'snail':
            train_dataset = synthetic.Snail(train = True)
            test_dataset = synthetic.Snail(train = False)
            self.nlabels = 0
        elif self.dataset == 'circle':
            train_dataset = synthetic.Circle(train = True)
            test_dataset = synthetic.Circle(train = False)
            self.nlabels = 0
        elif self.dataset == 'disc':
            train_dataset = synthetic.Disc(train = True)
            test_dataset = synthetic.Disc(train = False)
            self.nlabels = 0
        else:
            raise Exception("Dataset not found: " + dataset)

        if self.limit_train_size is not None:
            train_dataset = torch.utils.data.random_split(train_dataset, [self.limit_train_size, len(train_dataset)-self.limit_train_size])[0]

        self.train_loader = torch.utils.data.DataLoader(DatasetWithIndices(train_dataset, self.input_normalize_sym), batch_size=self.batch_size, shuffle=True, **self.dataloader_kwargs)
        self.test_loader = torch.utils.data.DataLoader(DatasetWithIndices(test_dataset, self.input_normalize_sym), batch_size=self.batch_size, shuffle=False, **self.dataloader_kwargs)

    def train(self): 
        self.setup_data_loaders()
        self.setup_trainers()

        self.global_iters = 0

        #epochs now configurable through num_iterations as well (for variable batch_size grid search)
        if self.epochs == None:
            iterations_per_epoch = int(len(self.train_loader.dataset) / self.batch_size)
            self.epochs = int(self.num_iterations/iterations_per_epoch)
        VIDEO_SIZE = 512
        with FFMPEG_VideoWriter('{}/{}.mp4'.format(self.viddir, self.prefix), (VIDEO_SIZE, VIDEO_SIZE), 3.0) as video:
            for self.epoch in range(self.epochs):
                for batch_idx, (x, y, idx) in enumerate(self.train_loader, start=0):
                    print(self.epoch, batch_idx, self.global_iters, len(x), len(self.train_loader))
                    self.global_iters += 1
                    batch = self.trainer.train_on_batch(x, idx, self.global_iters)

                    """
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
                    """
                    
                    if self.full_video or self.global_iters <= 1000 or self.global_iters % 1000 <= 100:
                        normalized_latents = self.normalize_latents(batch['video']['latents'])
                        frame = visual.draw_points(normalized_latents, VIDEO_SIZE, batch['video']['labels'], self.nlabels, int(self.global_iters / 1000))
                        video.write_frame(frame)
                    
                    if self.global_iters % self.log_interval == 0:                        
                        print("Global iter: {}, Train epoch: {}, batch: {}/{}, loss: {}".format(self.global_iters, self.epoch, batch_idx+1, len(self.train_loader), batch['loss']))
                        neptune.send_metric('train_loss', x=self.global_iters, y=batch['loss'])
                        #report global and local reg_losses
                        if self.trainer.trainer_type == 'local':
                            with torch.no_grad():
                                self.trainer.recalculate_latents()
                                pz_sample = self.trainer.sample_pz(len(self.train_loader.dataset)).to(self.device)
                                global_train_reg_loss = self.trainer.reg_loss_fn(self.trainer.x_latents, pz_sample.detach())   
                                neptune.send_metric('global_train_reg_loss', x=self.global_iters, y=global_train_reg_loss)
                                neptune.send_metric('local_train_reg_loss', x=self.global_iters, y=batch['reg_loss'])
                        else:
                            neptune.send_metric('global_train_reg_loss', x=self.global_iters, y=batch['reg_loss'])
                            
                        neptune.send_metric('train_rec_loss', x=self.global_iters, y=batch['rec_loss'])
                        neptune.send_metric('reg_lambda', x=self.global_iters, y=batch['reg_lambda'])
                        neptune.send_metric('blur-sigma', x=self.global_iters, y=batch['blur'])               

                        if self.global_iters % self.plot_interval == 0:
                            self.test()

        video.close()
        
    def plot_latent_2d(self, test_encode, test_targets, test_loss):
        if self.trainer.distribution == 'flower':
            test_encode = test_encode * 2 - 1.
        # save encoded samples plot
        plt.figure(figsize=(10, 10))
        #colordict = {0 : 'black', 1 : 'brown', 2 : 'b', 3 : 'cyan', 4 : 'g', 5 : 'lime', 6 : 'm', 7 : 'r', 8 : 'orange', 9 : 'y'}
        #for k in range(len(test_encode)):
        #    plt.scatter(test_encode[k, 0], test_encode[k, 1], c=colordict[test_targets[k]])
        plt.scatter(test_encode[:, 0], test_encode[:, 1], c=(10 * test_targets), cmap=plt.cm.Spectral)
        #plt.colorbar()
        plt.xlim([-1.5, 1.5])
        plt.ylim([-1.5, 1.5])
        plt.title('Test Latent Space\nLoss: {:.5f}'.format(test_loss))
        filename = '{}/test_latent_epoch_{}.pdf'.format(self.imagesdir, self.epoch + 1)
        plt.savefig(filename)        
        filename = '{}/test_latent_epoch_{}.png'.format(self.imagesdir, self.epoch + 1)
        plt.savefig(filename)
        plt.close()
        neptune.send_image('plot_latent_2d', x=self.global_iters, y=filename)

    def plot_images(self, x, train_rec, test_rec, gen):
        with torch.no_grad():
            plot_x, plot_train, plot_test, plot_gen = x, train_rec, test_rec, gen

            if self.input_normalize_sym:
                x_range = (-1., 1.)
            else:
                x_range = (0., 1.)
                
            utils.save_image(plot_x, 'test_samples', self.global_iters, '{}/test_samples_epoch_{}.png'.format(self.imagesdir, self.epoch + 1), x_range = x_range)
            utils.save_image(plot_train, 'train_reconstructions', self.global_iters, '{}/train_reconstructions_epoch_{}.png'.format(self.imagesdir, self.epoch + 1), normalize=True, x_range = x_range)
            utils.save_image(plot_test, 'test_reconstructions', self.global_iters, '{}/test_reconstructions_epoch_{}.png'.format(self.imagesdir, self.epoch + 1), normalize=True, x_range = x_range)
            utils.save_image(plot_gen, 'generated', self.global_iters, '{}/generated_epoch_{}.png'.format(self.imagesdir, self.epoch + 1), normalize=True, x_range = x_range)

    def plot_syn(self):
        reconstruction = torch.zeros(torch.Size([len(self.train_loader.dataset),2])).to(self.device).detach()
        generated = torch.zeros(torch.Size([len(self.train_loader.dataset),2])).to(self.device).detach()
        
        random_set = self.trainer.sample_pz(n=len(self.train_loader.dataset))
        original = torch.zeros(torch.Size([len(self.train_loader.dataset),2])).to(self.device).detach()
        
        for _, (x, _, idx) in enumerate(self.train_loader):
            with torch.no_grad():
                x = x.to(self.device)
                original[idx] = x
                reconstruction[idx], _ = self.model.forward(x)
                del x
                
                random_sample = random_set[idx]
                generated[idx] = self.trainer.decode_batch(random_sample)['decode']
                
        utils.save_scatter(reconstruction.detach().cpu().numpy()[:,0], reconstruction.detach().cpu().numpy()[:,1], self.imagesdir, 'train_reconstructions', self.global_iters)

        utils.save_scatter(generated.detach().cpu().numpy()[:,0], generated.detach().cpu().numpy()[:,1], self.imagesdir, 'generated', self.global_iters)
        
        utils.save_scatter(original.detach().cpu().numpy()[:,0], original.detach().cpu().numpy()[:,1], self.imagesdir, 'original', self.global_iters)

    def test(self):
        test_encode, test_targets, test_loss, test_reg_loss, test_rec_loss = list(), list(), 0.0, 0.0, 0.0
        full_test_encode = torch.zeros(torch.Size([len(self.test_loader.dataset),self.model.z_dim])).to(self.device).detach()
        
        with torch.no_grad():
            for _, (x_test, y_test, idx) in enumerate(self.test_loader, start=0):                
                test_evals = self.trainer.rec_loss_on_test(x_test)
                full_test_encode[idx] = test_evals['encode'].detach()
                test_encode.append(test_evals['encode'].detach())
                test_rec_loss += test_evals['rec_loss'].item()

                test_targets.append(y_test)
            
            normalized_latents = self.normalize_latents(full_test_encode)
            if self.distribution == 'gaussflower':
                covered = self.gaussflower_covered(full_test_encode)
            elif self.distribution == 'gaussimplex':
                covered = self.gaussimplex_covered(full_test_encode)
            else:
                covered = visual.covered_area(normalized_latents)
            test_rec_loss /= len(self.test_loader)
            test_reg_loss = self.trainer.reg_loss_on_test().item()
            test_loss = test_rec_loss + self.trainer.reg_lambda * test_reg_loss
        test_encode, test_targets = torch.cat(test_encode).cpu().numpy(), torch.cat(test_targets).cpu().numpy()

        if self.distribution == 'gaussflower':
            _, indices = self.split_to_petals(full_test_encode)
            labels_as_petals = np.zeros((10, 10))
            for i in range(10):
                for idx in indices[i]:
                    labels_as_petals[i][test_targets[idx]] += 1
            p = [i for i in range(10)]
            l = [j + 10 for j in range(10)]
            weighted_edges = [(i, j + 10, labels_as_petals[i][j]) for i in range(10) for j in range(10)]            
            B = nx.Graph()
            B.add_nodes_from(p, bipartite = 0)
            B.add_nodes_from(l, bipartite = 1)
            B.add_weighted_edges_from(weighted_edges)
            pairing = list(max_weight_matching(B, maxcardinality = True))
            pair_weight = 0
            for i in range(len(pairing)):
                if pairing[i][0] > pairing[i][1]:
                    pair_weight += labels_as_petals[pairing[i][1]][pairing[i][0] - 10]
                else:
                    pair_weight += labels_as_petals[pairing[i][0]][pairing[i][1] - 10]

        if self.distribution == 'gaussimplex':
            _, indices = self.split_to_vertex(full_test_encode)
            labels_as_vertex = np.zeros((10, 10))
            for i in range(10):
                for idx in indices[i]:
                    labels_as_vertex[i][test_targets[idx]] += 1
            p = [i for i in range(10)]
            l = [j + 10 for j in range(10)]
            weighted_edges = [(i, j + 10, labels_as_vertex[i][j]) for i in range(10) for j in range(10)]            
            B = nx.Graph()
            B.add_nodes_from(p, bipartite = 0)
            B.add_nodes_from(l, bipartite = 1)
            B.add_weighted_edges_from(weighted_edges)
            pairing = list(max_weight_matching(B, maxcardinality = True))
            pair_weight = 0
            for i in range(len(pairing)):
                if pairing[i][0] > pairing[i][1]:
                    pair_weight += labels_as_vertex[pairing[i][1]][pairing[i][0] - 10]
                else:
                    pair_weight += labels_as_vertex[pairing[i][0]][pairing[i][1] - 10]
                    
        neigh = NearestNeighbors(n_neighbors = 10)
        neigh.fit(test_encode)
        num_good_points = 0
        for k in range(len(test_encode)):
            nbrs = neigh.kneighbors(test_encode[k].reshape(1, -1), 10, return_distance = False)
            labels = list(test_targets[nbrs[0]])
            labels = set(labels)
            if len(labels) == 1:
                num_good_points += 1
        ratio = num_good_points / len(test_encode)

            #nat = self.trainer.sample_pz(len(self.train_loader.dataset)).to(self.device)
            #nat = nat.cpu().detach().numpy()
            #ratio_neighbor = visual.covered_neighborhood(test_encode, nat)

        #with open('ratio_{}_{}.txt'.format(self.trainer.trainer_type, self.trainer.reg_lambda), 'a') as file:
        #    file.write(str(ratio) + '\n')

        print('Test Epoch: {} ({:.2f}%)\tLoss: {:.6f}'.format(self.epoch + 1, float(self.epoch + 1) / (self.epochs) * 100., test_loss))
        neptune.send_metric('test_loss', x=self.global_iters, y=test_loss)
        neptune.send_metric('test_reg_loss', x=self.global_iters, y=test_reg_loss)
        neptune.send_metric('test_rec_loss', x=self.global_iters, y=test_rec_loss)
        neptune.send_metric('test_covered_area', x=self.global_iters, y=covered)
        neptune.send_metric('ratio_good_nn', x=self.global_iters, y=ratio)
        if self.distribution in ('gaussflower', 'gaussimplex'):
            neptune.send_metric('cluster_matching', x=self.global_iters, y=pair_weight)
            #neptune.send_metric('test_covered_neighbor', x=self.global_iters, y=ratio_neighbor)
        if len(test_targets.shape) == 2:
            test_targets = test_targets[:,self.trail_label_idx]
            
        self.plot_latent_2d(test_encode, test_targets, test_loss)
        
        if self.dataset in ('flower', 'square', 'disc', 'circle', 'snail'):
            self.plot_syn()
        else:
            with torch.no_grad():
                _, (x, _, _) = enumerate(self.test_loader, start=0).__next__()
                test_reconstruct = self.trainer.reconstruct(x)

                _, (x, _, _) = enumerate(self.train_loader, start=0).__next__()
                train_reconstruct = self.trainer.reconstruct(x)
                gen_batch = self.trainer.decode_batch(self.trainer.sample_pz(n=self.batch_size))
                self.plot_images(x, train_reconstruct, test_reconstruct, gen_batch['decode'])


def main(argv):
    gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_param, skip_unknown=True)
    op_config_str = gin.config._CONFIG

    use_neptune = "NEPTUNE_API_TOKEN" in os.environ
    if use_neptune:

        params = utils.get_gin_params_as_dict(gin.config._CONFIG)
        neptune.init(project_qualified_name="csadrian/global-autoencoders")

        exp = neptune.create_experiment(params=params, name="exp")
        #ONLY WORKS FOR ONE GIN-CONFIG FILE
        with open(FLAGS.gin_file[0]) as ginf:
            param = ginf.readline()
            while param:
                param = param.replace('.','-').replace('=','-').replace(' ','').replace('\'','').replace('\n','').replace('@','')
                #neptune.append_tag(param)
                param = ginf.readline()
        #for tag in opts['tags'].split(','):
        #  neptune.append_tag(tag)
    else:
        neptune.init('shared/onboarding', api_token='ANONYMOUS', backend=neptune.OfflineBackend())

    er = ExperimentRunner(prefix=exp.id)
    er.train()

    params = utils.get_gin_params_as_dict(gin.config._OPERATIVE_CONFIG)
    for k, v in params.items():
        neptune.set_property(k, v)
    neptune.stop()
    print('fin')

if __name__ == '__main__':
    flags.DEFINE_multi_string('gin_file', None, 'List of paths to the config files.')
    flags.DEFINE_multi_string('gin_param', None, 'Newline separated list of Gin parameter bindings.')
    FLAGS = flags.FLAGS
    app.run(main)
