import random
import sys
from typing import List

import numpy as np
import torch

from geomloss import SamplesLoss

from src.encoder_critic import EncoderCritic
from src.decoder_generator import DecoderGenerator
from src.data_loader import DataLoader


class TVAEGANNetwork:
    def __init__(self, in_size: int, hidden_sizes: List[int], latent_dim: int, cat_size: int, num_size: int,
                 w_regularize: float, w_reconstruct: float, s_generat: float, s_encoder: float, clip: float,
                 dropout: float, ot_loss: dict):

        self._set_seed()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(device)

        self.device = device
        self.dropout = dropout
        self.latent_dim = latent_dim
        self.w_regularize = w_regularize
        self.w_reconstruct = w_reconstruct
        self.s_generat = s_generat
        self.s_encoder = s_encoder
        self.clip = clip

        self.encoder = EncoderCritic(in_size=in_size, hidden_dims=hidden_sizes, out_size=latent_dim,
                                     dropout=self.dropout, device=self.device, is_vae=True).to(self.device)
        print(f"encoder total params {sum(p.numel() for p in self.encoder.parameters())}")
        print(f"encoder total trainable params {sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)}")

        self.generat = DecoderGenerator(in_size=latent_dim, hidden_dims=hidden_sizes[::-1], out_size=in_size,
                                        cat_size=cat_size, num_size=num_size,
                                        dropout=self.dropout, device=self.device).to(self.device)

        print(f"generat total params {sum(p.numel() for p in self.generat.parameters())}")
        print(f"generat total trainable params {sum(p.numel() for p in self.generat.parameters() if p.requires_grad)}")

        self.critic = EncoderCritic(in_size=in_size, hidden_dims=hidden_sizes, out_size=1,
                                    dropout=self.dropout, device=self.device).to(self.device)

        print(f"critic total params {sum(p.numel() for p in self.critic.parameters())}")
        print(f"critic total trainable params {sum(p.numel() for p in self.critic.parameters() if p.requires_grad)}")

        self.encoder_optim: torch.optim.RMSprop = None
        self.decoder_optim: torch.optim.RMSprop = None
        self.generat_optim: torch.optim.RMSprop = None
        self.critic_optim: torch.optim.RMSprop = None

        self.loss_regularize = SamplesLoss(**ot_loss)
        self.loss_restore = torch.nn.MSELoss()
        self.loss_critic = torch.nn.MSELoss()

    def _set_seed(self, seed: int = 42):
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

    def train(self, data_loader: DataLoader, epochs: int,  lr_encoder: float, lr_critic: float, lr_generat: float):
        self.encoder_optim = torch.optim.RMSprop(self.encoder.parameters(), lr=lr_encoder)
        self.decoder_optim = torch.optim.RMSprop(self.generat.parameters(), lr=lr_encoder)
        self.generat_optim = torch.optim.RMSprop(self.generat.parameters(), lr=lr_generat)
        self.critic_optim = torch.optim.RMSprop(self.critic.parameters(), lr=lr_critic)

        self.encoder.train(True)
        self.generat.train(True)
        self.critic.train(True)

        self._train(epochs, data_loader)

        self.encoder.train(False)
        self.generat.train(False)
        self.critic.train(False)

    def _train(self, num_epochs: int, data_loader: DataLoader):
        for epoch in range(num_epochs):
            encoder_losses = []
            critic_losses = []
            generat_losses = []

            for step, rows in enumerate(data_loader):
                batch_size = rows[0].shape[0]
                x = rows[0]
                x = x.type(torch.FloatTensor).to(self.device)

                if self.s_generat != sys.maxsize:
                    critic_loss = self._critic_step(batch_size, x)
                    critic_losses.append(critic_loss)

                if step % self.s_generat == 0 and self.s_generat != sys.maxsize:
                    generat_loss = self._generat_step(batch_size, x)
                    generat_losses.append(generat_loss)

                if step % self.s_encoder == 0 and self.s_encoder != sys.maxsize:
                    encoder_loss = self._encoder_step(batch_size, x)
                    encoder_losses.append(encoder_loss)

            # ===================log========================
            encoder_loss_avg = sum(encoder_losses) / len(encoder_losses) if len(encoder_losses) > 0 else 0
            critic_loss_avg = sum(critic_losses) / len(critic_losses) if len(critic_losses) > 0 else 0
            generat_loss_avg = sum(generat_losses) / len(generat_losses) if len(generat_losses) > 0 else 0

            metrics = {
                "epoch": epoch + 1,
                "encoder_loss_avg": encoder_loss_avg,
                "critic_loss_avg": critic_loss_avg,
                "generat_loss_avg": generat_loss_avg,
            }
            print(metrics)

    def _encoder_step(self, batch_size, x):
        self.encoder_optim.zero_grad()
        self.decoder_optim.zero_grad()
        self.generat_optim.zero_grad()
        self.critic_optim.zero_grad()

        # ======================encoder forward========================
        x_encoded = self.encoder(x)
        y = self.generat(x_encoded)
        z = torch.randn(x_encoded.shape).to(self.device)

        # ======================encoder loss===========================
        regularize_loss = self.loss_regularize(x_encoded, z)
        reconstruc_loss = self.loss_restore(y, x)

        loss = (regularize_loss * self.w_regularize) + (reconstruc_loss * self.w_reconstruct)

        # ===================backward====================
        loss.backward()
        self.encoder_optim.step()
        self.decoder_optim.step()

        return loss.item()

    def _generat_step(self, batch_size, x):
        self.encoder_optim.zero_grad()
        self.decoder_optim.zero_grad()
        self.generat_optim.zero_grad()
        self.critic_optim.zero_grad()

        # ===================generator forward==========================
        z = torch.randn((batch_size, self.latent_dim)).to(self.device)
        z_out = self.generat(z)
        critic_z_out = self.critic(z_out)

        # ===================generator loss=============================
        loss = -torch.mean(critic_z_out)

        # ===================generator backward=========================
        loss.backward()
        self.generat_optim.step()

        return loss.item()

    def _critic_step(self, batch_size, x):
        self.encoder_optim.zero_grad()
        self.decoder_optim.zero_grad()
        self.generat_optim.zero_grad()
        self.critic_optim.zero_grad()

        # ==============critic forward =========================
        critic_x_out = self.critic(x)

        with torch.no_grad():
            z = torch.randn((batch_size, self.latent_dim)).to(self.device)
            z_out = self.generat(z)
        critic_z_out = self.critic(z_out)

        with torch.no_grad():
            x_encoded = self.encoder(x)
            y_out = self.generat(x_encoded)
        critic_y_out = self.critic(y_out)

        # ==============critic loss ============================
        loss = -2*torch.mean(critic_x_out) + torch.mean(critic_y_out) + torch.mean(critic_z_out)

        # ==============critic backward ========================
        loss.backward()
        self.critic_optim.step()
        for p in self.critic.parameters():
            p.data.clamp_(-self.clip, self.clip)

        return loss.item()

    def generate(self, z):
        with torch.no_grad():
            z = torch.from_numpy(z).type(torch.FloatTensor).to(self.device)
            z_out = self.generat(z)

            out = z_out.detach().cpu().numpy()

        return out

    def save(self, path):
        torch.save(
            {
                "encoder_state_dict": self.encoder.state_dict(),
                "decoder_state_dict": self.generat.state_dict(),
                "critic_state_dict": self.critic.state_dict(),
            },
            path,
        )

    def load(self, path):
        checkpoint = torch.load(path)
        self.encoder.load_state_dict(checkpoint["encoder_state_dict"])
        self.generat.load_state_dict(checkpoint["decoder_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
