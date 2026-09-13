import io
from copy import deepcopy
from typing import List
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

import torch

from src.data_handler import DataHandler
from src.data_loader import DataLoader
from src.tvaegan_network import TVAEGANNetwork


class TVAEGANSynthesizer(BaseEstimator):
    def __init__(self, epochs: int = 700, batch_size: int = 500, cat_emb_size: int = 25, num_emb_size: int = 25,
                 w_regularize: float = 1, w_reconstruct: float = 10, s_generat: int = 5, s_encoder: int = 5,
                 lr_generat: float = 0.00005, lr_critic: float = 0.00005, lr_encoder: float = 0.00005,
                 clip: float = 0.01, dropout: float = 0.1, ot_loss: dict = {"loss": "energy"},
                 hidden_layers_multipliers: List[int] = [1, 1],
                 shuffle: bool = True, seed: int = 42):
        super().__init__()

        self.epochs : int = epochs
        self.batch_size : int = batch_size
        self.lr_critic : float = lr_critic
        self.lr_encoder : float = lr_encoder
        self.lr_generat : float = lr_generat

        self.cat_emb_size : int = cat_emb_size
        self.num_emb_size : int = num_emb_size

        self.w_regularize : float = w_regularize
        self.w_reconstruct : float = w_reconstruct
        self.s_encoder : int = s_encoder
        self.s_generat : int = s_generat
        self.clip : float = clip
        self.dropout : float = dropout
        self.shuffle : bool = shuffle
        self.ot_loss : dict = ot_loss
        self.hidden_layers_multipliers : List[int] = hidden_layers_multipliers
        self.seed : int = seed

        self.latent_dim: int = None # type: ignore
        self.data_handler: DataHandler = None # type: ignore
        self.model: TVAEGANNetwork = None # type: ignore

    def fit(self, df: pd.DataFrame):
        self.data_handler = DataHandler()
        scaled_np = self.data_handler.transform(df)
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

        self.model = TVAEGANNetwork(
            in_size=in_size, 
            hidden_sizes=hidden_layers_sizes, 
            latent_dim=self.latent_dim,
            cat_size=sum(cat_sizes), 
            num_size=len(num_sizes),
            w_regularize=self.w_regularize, 
            w_reconstruct=self.w_reconstruct,
            s_generat=self.s_generat, 
            s_encoder=self.s_encoder, 
            clip=self.clip,
            dropout=self.dropout, 
            ot_loss=self.ot_loss, 
            seed=self.seed
        )

        torch_ds = torch.from_numpy(scaled_np)
        data_loader = DataLoader(torch_ds, batch_size=self.batch_size, shuffle=self.shuffle)

        # do training
        self.model.train(
            data_loader=data_loader, 
            epochs=self.epochs,
            lr_encoder=self.lr_encoder, 
            lr_critic=self.lr_critic, 
            lr_generat=self.lr_generat
        )

        return self

    def predict(self, samples: int, seed: int | None = None):
        if seed is not None:
            np.random.seed(seed)

        X = np.random.randn(samples, self.latent_dim)
        Z = self.model.generate(X)
        synth_df = self.data_handler.reverse(Z)

        return synth_df

    def save(self, path):
        model_buffer = io.BytesIO()
        self.model.save(model_buffer)

        torch.save(
            {
                "params": self.get_params(), 
                "data_handler": self.data_handler,
                "model": model_buffer.getvalue()
            }, 
            path
        )

    @classmethod
    def load(cls, path):
        # weights_only=False unpickles arbitrary objects - only load trusted files
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)

        synthesizer = cls(**checkpoint["params"])
        synthesizer.data_handler = checkpoint["data_handler"]

        # rebuild the network with the same sizes fit() used
        cat_sizes = synthesizer.data_handler.get_cat_sizes()
        num_sizes = synthesizer.data_handler.get_num_sizes()
        cat_size_sum = sum([synthesizer.cat_emb_size for cat_size in cat_sizes])
        num_size_sum = sum([min(synthesizer.num_emb_size, num_size) for num_size in num_sizes])
        vec_total_length = cat_size_sum + num_size_sum
        in_size = sum(cat_sizes) + len(num_sizes)
        hidden_layers_sizes = [int(vec_total_length * hidden_layer_multiplier)
                               for hidden_layer_multiplier in synthesizer.hidden_layers_multipliers]
        synthesizer.latent_dim = len(synthesizer.data_handler._columns)

        synthesizer.model = TVAEGANNetwork(
            in_size=in_size, 
            hidden_sizes=hidden_layers_sizes,
            latent_dim=synthesizer.latent_dim,
            cat_size=sum(cat_sizes), 
            num_size=len(num_sizes),
            w_regularize=synthesizer.w_regularize,
            w_reconstruct=synthesizer.w_reconstruct,
            s_generat=synthesizer.s_generat, 
            s_encoder=synthesizer.s_encoder,
            clip=synthesizer.clip, 
            dropout=synthesizer.dropout,
            ot_loss=synthesizer.ot_loss, 
            seed=synthesizer.seed
        )
        synthesizer.model.load(io.BytesIO(checkpoint["model"]))

        return synthesizer
