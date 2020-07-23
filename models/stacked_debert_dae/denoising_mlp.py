import torch
import torch.nn as nn
import numpy as np


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder_layer1 = nn.Sequential(
            nn.Linear(768, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
        )
        self.encoder_layer2 = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 24), nn.ReLU(),
            nn.Linear(24, 12)
        )
        self.decoder_layer1 = nn.Sequential(
            nn.Linear(12, 24), nn.ReLU(),
            nn.Linear(24, 32), nn.ReLU(),
            nn.Linear(32, 64), nn.ReLU(),
        )
        self.decoder_layer2 = nn.Sequential(
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, 768), nn.Tanh()
        )

    def forward(self, x):
        out_encoder = self.encoder_layer1(x)
        # print(np.shape(out_encoder))
        out_encoder = self.encoder_layer2(out_encoder)
        # print(np.shape(out_encoder))
        out_decoder = self.decoder_layer1(out_encoder)
        # print(np.shape(out_decoder))
        out_decoder = self.decoder_layer2(out_decoder)
        # print(np.shape(out_decoder))
        return out_encoder, out_decoder

"""
class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder_layer1 = nn.Sequential(
            nn.Linear(768, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
        )
        self.encoder_layer2 = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
        )
        self.encoder_layer3 = nn.Sequential(
            nn.Linear(32, 24), nn.ReLU(),
            nn.Linear(24, 12)
        )
        self.decoder_layer1 = nn.Sequential(
            nn.Linear(12, 24), nn.ReLU(),
            nn.Linear(24, 32), nn.ReLU(),
        )
        self.decoder_layer2 = nn.Sequential(
            nn.Linear(32, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
        )
        self.decoder_layer3 = nn.Sequential(
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, 768), nn.Tanh()
        )

    def forward(self, x):
        out_encoder = self.encoder_layer1(x)
        # print(np.shape(out_encoder))
        out_encoder = self.encoder_layer2(out_encoder)
        out_encoder = self.encoder_layer3(out_encoder)
        # print(np.shape(out_encoder))
        out_decoder = self.decoder_layer1(out_encoder)
        # print(np.shape(out_decoder))
        out_decoder = self.decoder_layer2(out_decoder)
        out_decoder = self.decoder_layer3(out_decoder)
        # print(np.shape(out_decoder))
        return out_encoder, out_decoder
"""
