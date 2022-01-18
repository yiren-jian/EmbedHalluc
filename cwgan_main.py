import argparse
import os
import numpy as np
import math
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets

from cwgan_model import Generator, Discriminator
import json

from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset


def sample_image(n_row, batches_done):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    # Sample noise
    z = Tensor(np.random.normal(0, 1, (n_row ** 2, opt.latent_dim)))
    # Get labels ranging from 0 to n_classes for n rows
    labels = np.array([num for _ in range(n_row) for num in range(n_row)])
    with torch.no_grad():
        labels = LongTensor(labels)
        gen_imgs = generator(z, labels)
    save_image(gen_imgs.data, "images/%d.png" % batches_done, nrow=n_row, normalize=True)


def compute_gradient_penalty(D, real_samples, fake_samples, labels):
    """Calculates the gradient penalty loss for WGAN GP.
       Warning: It doesn't compute the gradient w.r.t the labels, only w.r.t
       the interpolated real and fake samples, as in the WGAN GP paper.
    """
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1)))
    labels = LongTensor(labels)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates, labels)
    fake = Tensor(real_samples.shape[0], 1).fill_(1.0)
    fake.requires_grad = False
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )
    gradients = gradients[0].view(gradients[0].size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


# ----------
#  Training
# ----------

class EmbeddingDataset(Dataset):
    def __init__(self, opt):
        with open('cache_embeddings/%s_%s_%s_%s.json'%(opt.model_name_or_path, opt.few_shot_type, opt.task_name, opt.seed)) as json_file:
            data = json.load(json_file)
        self.embeddings = []
        self.labels = []
        for i in data:
            for emb_i in range(len(data[i])):
                self.embeddings.append(torch.tensor(data[i][emb_i]).squeeze(0))
                self.labels.append(torch.tensor(int(i)).long())

        self.num_classes = len(data)
        self.data_len = len(self.labels)

    def __getitem__(self, index):
        return self.embeddings[index].float(), self.labels[index]

    def __len__(self):
        return self.data_len


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=150, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=1, help="number of image channels")
    parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
    parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
    parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
    parser.add_argument("--dataset", type=str, choices=['mnist', 'fashion'],
                        default='mnist', help="dataset to use")
    # model_args.model_name_or_path, model_args.few_shot_type, data_args.task_name, training_args.seed)
    parser.add_argument("--model_name_or_path", type=str, default='roberta-large', help="roberta-larger | bert-large-cased")
    parser.add_argument("--few_shot_type", type=str, default='finetune', help="finetune | prompt")
    parser.add_argument("--task_name", type=str, default='sst-5', help="sst-5 | cola | mrpc | ...")
    parser.add_argument("--seed", type=int, default=13, help="13 | 21 | 42 | 87 | 100")
    parser.add_argument("--seq_len", type=int, default=128, help="input length for Transformer, i.e. 128, 256, 512")

    opt = parser.parse_args()
    print(opt)

    embeddings_set = EmbeddingDataset(opt)
    dataloader = DataLoader(embeddings_set, batch_size=8, shuffle=True)
    emb, lbl = embeddings_set.__getitem__(0)

    opt.n_classes=embeddings_set.num_classes
    opt.img_shape = emb.shape

    cuda = True if torch.cuda.is_available() else False

    # Loss weight for gradient penalty
    lambda_gp = 100   #### original default is 10

    # Initialize generator and discriminator
    generator = Generator(opt)
    discriminator = Discriminator(opt)

    if cuda:
        generator.cuda()
        discriminator.cuda()


    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

    batches_done = 0
    for epoch in range(opt.n_epochs):
        for i, (embs, labels) in enumerate(dataloader):
            batch_size = embs.shape[0]

            # Move to GPU if necessary
            real_embs = embs.type(Tensor)
            labels = labels.type(LongTensor)

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Sample noise and labels as generator input
            z = Tensor(np.random.normal(0, 1, (embs.shape[0], opt.latent_dim)))

            # Generate a batch of embeddings
            fake_embs = generator(z, labels)

            # Real embeddings
            real_validity = discriminator(real_embs, labels)
            # Fake embeddings
            fake_validity = discriminator(fake_embs, labels)
            # Gradient penalty
            gradient_penalty = compute_gradient_penalty(
                                discriminator, real_embs.data, fake_embs.data,
                                labels.data)
            # Adversarial loss
            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty

            d_loss.backward()
            optimizer_D.step()

            optimizer_G.zero_grad()

            # Train the generator every n_critic steps
            if i % opt.n_critic == 0:

                # -----------------
                #  Train Generator
                # -----------------

                # Generate a batch of embeddings
                fake_embs = generator(z, labels)
                # Loss measures generator's ability to fool the discriminator
                # Train on fake embeddings
                fake_validity = discriminator(fake_embs, labels)
                g_loss = -torch.mean(fake_validity)

                g_loss.backward()
                optimizer_G.step()

                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
                )

                batches_done += opt.n_critic

    if not os.path.exists('cache_generators'):
        os.mkdir('cache_generators')
    torch.save(generator.state_dict(), 'cache_generators/%s_%s_%s_%s.pth'%(opt.model_name_or_path, opt.few_shot_type, opt.task_name, opt.seed))
