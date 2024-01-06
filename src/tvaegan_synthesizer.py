from copy import deepcopy
from typing import List
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

import torch

from data_handler import DataHandler
from data_loader import DataLoader
from tvaegan_network import TVAEGANNetwork


class TVAEGANSynthesizer(BaseEstimator):
    def __init__(self, epochs: int = 700, batch_size: int = 500, cat_emb_size: int = 25, num_emb_size: int = 25,
                 w_regularize: float = 1, w_reconstruct: float = 10, s_generat: int = 5, s_encoder: int = 5,
                 lr_generat: float = 0.00005, lr_critic: float = 0.00005, lr_encoder: float = 0.00005,
                 clip: float = 0.01, dropout: float = 0.1, ot_loss: dict = {"loss": "energy"},
                 hidden_layers_multiplers: List[int] = [1, 1],
                 shuffle: bool = True):
        super().__init__()

        self.epochs = epochs
        self.batch_size = batch_size
        self.lr_critic = lr_critic
        self.lr_encoder = lr_encoder
        self.lr_generat = lr_generat

        self.cat_emb_size = cat_emb_size
        self.num_emb_size = num_emb_size

        self.w_regularize = w_regularize
        self.w_reconstruct = w_reconstruct
        self.s_encoder = s_encoder
        self.s_generat = s_generat
        self.clip = clip
        self.dropout = dropout
        self.shuffle = shuffle
        self.ot_loss = ot_loss
        self.hidden_layers_multipliers = hidden_layers_multiplers

        self.latent_dim: int = None
        self.data_handler: DataHandler = None
        self.model: TVAEGANNetwork = None

    def fit(self, df: pd.DataFrame):
        self.data_handler = DataHandler(df)
        scaled_np = self.data_handler.transform()
        df_reversed = self.data_handler.reverse(deepcopy(scaled_np))
        print(df.compare(df_reversed))

        cat_sizes = self.data_handler.get_cat_sizes()
        num_sizes = self.data_handler.get_num_sizes()
        cat_size_sum = sum([self.cat_emb_size for cat_size in cat_sizes])
        num_size_sum = sum([min(self.num_emb_size, num_size) for num_size in num_sizes])

        vec_total_length = cat_size_sum + num_size_sum
        print(f"vec_total_length: {vec_total_length}")

        in_size = sum(cat_sizes) + len(num_sizes)
        print(f"in_size: {in_size}")

        hidden_layers_sizes = [int(vec_total_length * hidden_layer_multiplier)
                               for hidden_layer_multiplier in self.hidden_layers_multipliers]
        print(f"hidden_layers_sizes: {hidden_layers_sizes}")

        self.latent_dim = len(df.columns)
        print(f"latent_dim: {self.latent_dim}")

        self.model = TVAEGANNetwork(in_size=in_size, hidden_sizes=hidden_layers_sizes, latent_dim=self.latent_dim,
                                    cat_size=sum(cat_sizes), num_size=len(num_sizes),
                                    w_regularize=self.w_regularize, w_reconstruct=self.w_reconstruct,
                                    s_generat=self.s_generat, s_encoder=self.s_encoder, clip=self.clip,
                                    dropout=self.dropout, ot_loss=self.ot_loss)

        torch_ds = torch.from_numpy(scaled_np)
        data_loader = DataLoader(torch_ds, batch_size=self.batch_size, shuffle=self.shuffle)

        # do training
        self.model.train(data_loader=data_loader, epochs=self.epochs,
                         lr_encoder=self.lr_encoder, lr_critic=self.lr_critic, lr_generat=self.lr_generat)

        return self

    def predict(self, samples: int):
        X = np.random.randn(samples, self.latent_dim)
        Z = self.model.generate(X)
        synth_df = self.data_handler.reverse(Z)

        return synth_df
