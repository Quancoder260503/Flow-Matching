import cv2
import numpy as np 
import tqdm 
import torch.nn.functional as F 
import logging 
import torch 
import os 
import shutil 
import tensorboardX
import torch.optim as optim
from torchvision.datasets import MNIST, CIFAR10, CIFAR100, LSUN
import torchvision.transforms as transforms 
from torch.utils.data import DataLoader, Subset 
from torchvision.utils import save_image, make_grid 
from PIL import Image 
from Unet import Unet
from EMA import EMAHelper
import matplotlib.animation as animation
import matplotlib.pyplot as plt 
import glob
from Scheduler import ConditionalOptimalTransportScheduler 
from ProbabilityPath import AffineProbabilityPath


def skewed_timestep_sample(num_samples : int, device : torch.device) -> torch.Tensor: 
    P_mean = -1.2 
    P_std  = 1.2 

    rnd_normal = torch.randn((num_samples, ), device = device) 
    sigma = (rnd_normal * P_std + P_mean).exp()
    time = 1 / (1 + sigma)
    time = torch.clip(time, min = 0.0001, max = 1.0) 
    return time 


class Runner(object):
    def __init__(self, args): 
        self.args = args    
        self.prepare_data()

        self.model = Unet(
            image_size                   = self.args.image_size, 
            in_channels                  = self.args.in_channels, 
            model_channels               = self.args.model_channels, 
            out_channels                 = self.args.out_channels,
            num_residual_blocks          = self.args.num_residual_blocks, 
            attention_resolutions        = self.args.attention_resolution, 
            dropout                      = self.args.dropout, 
            channel_mult                 = self.args.channel_mult, 
            conv_resample                = self.args.conv_resample,
            dims                         = self.args.dims, 
            num_classes                  = self.args.num_classes, 
            num_attention_heads          = self.args.num_attention_heads, 
            use_scale_shift_norm         = self.args.use_scale_shift_norm,
            residual_block_up_down       = self.args.residual_block_up_down, 
            embedding_to_model_dim_ratio = self.args.embedding_to_model_dim_ratio, 
        )
        

        self.prob_path = AffineProbabilityPath(scheduler = ConditionalOptimalTransportScheduler())
        
        print('=' * 40)
        print("TOTAL PARAMS:")
        print(sum(p.numel() for p in self.model.parameters() if p.requires_grad))

        self.model = self.model.to(self.args.device)
        self.optimizer    = self.get_optimizer(model_params = self.model.parameters())
        self.lr_scheduler = self.get_lr_scheduler()
        
        print('=' * 40)
        print('LOAD EMA HELPER')
        self.ema_helper = EMAHelper(mu = self.args.ema_rate) 
        self.ema_helper.register(self.model)

        # For evaluation
        
        self.start_epoch = 0
        self.step = 0
        
        if self.args.resume_training : 
            states = torch.load(os.path.join(self.args.path, 'checkpoint', f'{self.args.model_version}.pth'))
            self.model.load_state_dict(states[0])
            self.optimizer.load_state_dict(states[1])
            self.lr_scheduler.load_state_dict(states[2])
            self.start_epoch = states[3]
            self.step = states[4]
            self.ema_helper.load_state_dict(states[5])

        self.model = self.model.to(self.args.device)

    def get_lr_scheduler(self):
        if self.args.lr_scheduler == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer = self.optimizer, 
                step_size = self.args.lr_step_size, 
                gamma     = self.args.lr_gamma, 
            )
            return scheduler
        elif self.args.lr_scheduler == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer  = self.optimizer,
                T_max      = self.args.num_training_epochs, 
                eta_min    = self.args.lr_eta_min, 
                last_epoch = -1, 
            )
            return scheduler
        else : 
            raise NotImplementedError('Learning Rate Scheduler not supported for now') 

    def get_optimizer(self, model_params): 
        if self.args.optimizer == 'Adam': 
            return optim.Adam(
                params       = model_params, 
                lr           = self.args.learning_rate, 
                weight_decay = self.args.weight_decay,
                betas        = self.args.betas, 
            )
        
        elif self.args.optimizer == 'RMSProp': 
            return optim.RMSprop(
                params       = model_params, 
                lr           = self.args.learning_rate, 
                weight_decay = self.args.weight_decay, 
            )
        
        elif self.args.optimizer == 'SGD': 
            return optim.SGD(
                params       = model_params, 
                lr           = self.args.learning_rate, 
                weight_decay = self.args.weight_decay, 
            )
        else : 
            raise NotImplementedError('Optimizer not supported')
        
    def prepare_data(self):
        cifar_normalize  = transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5])
        mnist_normalize  = transforms.Normalize(mean = [0.5],           std = [0.5])

        def make_transforms(image_size, normalize, random_flip = False):
            base = [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                normalize,
            ]
            if random_flip:
                base.insert(1, transforms.RandomHorizontalFlip(p = 0.5))
            return transforms.Compose(base)

        self.inverse_data_transform = transforms.Compose([
            transforms.Normalize(mean = [-1.0], std = [2.0]),
            transforms.Resize((self.args.image_size, self.args.image_size)),
            transforms.ToPILImage(),
        ])

        if self.args.dataset in ('CIFAR10', 'CIFAR100'):
            is_cifar100 = self.args.dataset == 'CIFAR100'
            DatasetClass = CIFAR100 if is_cifar100 else CIFAR10
            cifar_root   = os.path.join(self.args.path, 'datasets', 'cifar100' if is_cifar100 else 'cifar10')

            download_dataset = self.args.download
            if os.path.isdir(cifar_root) and os.listdir(cifar_root):
                download_dataset = False
            elif not self.args.download:
                raise RuntimeError(
                    f"{self.args.dataset} dataset directory '{cifar_root}' is missing or empty and download is disabled."
                )
            train_dataset = DatasetClass(
                root      = cifar_root,
                train     = True,
                download  = download_dataset,
                transform = make_transforms(self.args.image_size, cifar_normalize, self.args.random_flip),
            )
            test_dataset = DatasetClass(
                root      = cifar_root,
                train     = False,
                download  = download_dataset,
                transform = make_transforms(self.args.image_size, cifar_normalize),
            )
        elif self.args.dataset == 'MNIST':
            mnist_root = os.path.join(self.args.path, 'datasets', 'mnist')
            download_dataset = self.args.download
            if os.path.isdir(mnist_root) and os.listdir(mnist_root):
                download_dataset = False
            elif not self.args.download:
                raise RuntimeError(
                    f"MNIST dataset directory '{mnist_root}' is missing or empty and download is disabled."
                )
            train_dataset = MNIST(
                root      = mnist_root,
                train     = True,
                download  = download_dataset,
                transform = make_transforms(self.args.image_size, mnist_normalize, self.args.random_flip),
            )
            test_dataset = MNIST(
                root      = mnist_root,
                train     = False,
                download  = download_dataset,
                transform = make_transforms(self.args.image_size, mnist_normalize),
            )
        else:
            raise NotImplementedError('Dataset not supported')

        self.train_loader = DataLoader(
            dataset     = train_dataset,
            batch_size  = self.args.batch_size,
            shuffle     = True,
            pin_memory  = True,
            num_workers = self.args.num_workers,
        )
        self.test_loader = DataLoader(
            dataset     = test_dataset,
            batch_size  = self.args.batch_size,
            pin_memory  = True,
            num_workers = self.args.num_workers,
        )

    def train(self):
        self.train_vector_field()

    def train_vector_field(self):
        tensorboard_path = os.path.join(self.args.path, 'tensorboard', self.args.doc)
        if os.path.exists(tensorboard_path): 
           shutil.rmtree(tensorboard_path)
        
        tensorboard_logger = tensorboardX.SummaryWriter(log_dir = tensorboard_path)


        test_iter = iter(self.test_loader)
        print('=' * 40)
        print("START TRAINING VECTOR FIELD:")

        for epoch in range(self.start_epoch, self.args.num_training_epochs): 
            losses = [] 
            for index, (img, _) in enumerate(self.train_loader): 
                self.step += 1 

                self.model.train()

                batch_size = img.shape[0]
                img = img.to(self.args.device) 

                if self.args.skewed_timesteps : 
                    t = skewed_timestep_sample(batch_size, device = self.args.device)
                else : 
                    t = torch.rand(batch_size).to(self.args.device)

                noise = torch.randn_like(img).to(self.args.device)

                path_sample = self.prob_path.sample(t = t, x_0 = noise, x_1 = img)
                x_t = path_sample.x_t 
                dx_t = path_sample.dx_t


                loss = torch.pow(self.model(x_t, t) - dx_t, 2).mean()


                losses.append(loss.item())
                loss.backward()

                if self.args.max_grad > 0 : 
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad)

                self.optimizer.step() 
                self.lr_scheduler.step()
                self.ema_helper.update(self.model)  

                tensorboard_logger.add_scalar('loss', loss.item(), global_step = self.step) 
                logging.info(f"step : {self.step}, loss : {loss.item()}")
        

                if self.step % 100 == 0:
                    self.model.eval()
                    try:
                        test_img, _ = next(test_iter)
                    except StopIteration:
                        test_iter = iter(self.test_loader)
                        test_img, _ = next(test_iter)

                    test_img = test_img.to(self.args.device) 

                    with torch.no_grad(): 
                        if self.args.skewed_timesteps : 
                            t = skewed_timestep_sample(test_img.shape[0], device = self.args.device)
                        else : 
                            t = torch.rand(test_img.shape[0]).to(self.args.device)

                        noise = torch.randn_like(test_img).to(self.args.device)
                        path_sample = self.prob_path.sample(t = t, x_0 = noise, x_1 = test_img)
                        x_t = path_sample.x_t 
                        dx_t = path_sample.dx_t
                        test_loss = torch.pow(self.model(x_t, t) - dx_t, 2).mean()
                        tensorboard_logger.add_scalar('test_loss', test_loss, global_step = self.step)
                        print(f"step : {self.step}, test_loss : {test_loss.item()}")

                if self.step % self.args.snapshot_freq == 0:
                    states = [
                        self.model.state_dict(),
                        self.optimizer.state_dict(),
                        self.lr_scheduler.state_dict(),
                        epoch, 
                        self.step, 
                        self.ema_helper.state_dict(), 
                    ]
                    torch.save(states, os.path.join(self.args.path, 'checkpoint',f'{self.args.model_version}.pth'))
                
            print(f"Epoch : {epoch}, step : {self.step}, Training Loss : {np.mean(losses)}")
    
    
    @torch.no_grad()
    def EulerSampler(self, shape):
        x = torch.randn(shape).to(self.args.device)
        dt = 1.0 / self.args.num_sampling_steps
        samples = [x]
        for i in range(self.args.num_sampling_steps):
            t = torch.full((shape[0], ), i * dt, device = self.args.device)
            x = x + self.model(x, t) * dt
            samples.append(x)
        return x, samples

    @torch.no_grad()
    def MidPointSampler(self, shape):
        x = torch.randn(shape).to(self.args.device)
        dt = 1.0 / self.args.num_sampling_steps
        samples = [x]
        for i in range(self.args.num_sampling_steps):
            t = torch.full((shape[0], ), i * dt, device = self.args.device)
            x_mid = x + self.model(x, t) * (dt / 2.0)
            x = x + self.model(x_mid, t + dt / 2.0) * dt
            samples.append(x)
        return x, samples

    @torch.no_grad()
    def HeunSampler(self, shape):
        x = torch.randn(shape).to(self.args.device)
        dt = 1.0 / self.args.num_sampling_steps
        samples = [x]
        for i in range(self.args.num_sampling_steps):
            t = torch.full((shape[0], ), i * dt, device=self.args.device)
            v0 = self.model(x, t)
            x_euler = x + v0 * dt
            v1 = self.model(x_euler, t + dt)
            x = x + dt * (v0 + v1) / 2.0
            samples.append(x)
        return x, samples
    
    def get_ode_solver(self):
        if self.args.sampler == 'euler':
            return self.EulerSampler
        elif self.args.sampler == 'heun':
            return self.HeunSampler
        elif self.args.sampler == 'mid_point':
            return self.MidPointSampler

    @torch.no_grad()
    def sample(self):
        os.makedirs(self.args.image_folder, exist_ok = True)
        img_id = 0
        num_samples = self.args.num_samples
        num_rounds = (num_samples - img_id) // self.args.sampling_batch_size

        self.sampler = self.get_ode_solver()

        with torch.no_grad():
            for _ in tqdm.tqdm(range(num_rounds), desc = 'Generating img for evaluation'):
                batch_size = self.args.sampling_batch_size

                shape = (batch_size, self.args.in_channels, self.args.image_size, self.args.image_size)
                out, intermediate_samples = self.sampler(shape)

                for i in range(batch_size):
                    img = self.inverse_data_transform(out[i])

                    fig, ax = plt.subplots()
                    ax.axis('off')
                    frames = []

                    for t, sample in enumerate(intermediate_samples):
                        noise_img = self.inverse_data_transform(sample[i].cpu())
                        np_noise = np.array(noise_img)

                        artist = ax.imshow(np_noise, animated = True)
                        label  = ax.text(
                            0.5, -0.05, 
                            f"t = {1.0 * t / self.args.num_sampling_steps}",
                            transform = ax.transAxes, 
                            ha = 'center', 
                            color = 'gray', 
                        )
                        frames.append([artist, label])

                    animate = animation.ArtistAnimation(
                        fig,
                        frames,
                        interval = 50,
                        blit = True,
                        repeat_delay = 1000, 
                    )
                    animate.save(
                        os.path.join(self.args.image_folder, f'Flow_Matching_{img_id}.gif'),
                        writer = 'pillow'
                    )
                    plt.close(fig)  


                    np_img = np.array(img)
                    if np_img.ndim == 3 and np_img.shape[2] == 3:
                        np_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(os.path.join(self.args.image_folder, f'{img_id}.png'), np_img)
                    img_id += 1
    
        

