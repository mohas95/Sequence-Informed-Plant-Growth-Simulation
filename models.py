import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
import torchvision.models as models
from torch.utils.data import random_split
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from tqdm import tqdm
import sys

import requests
import os
# import json
import zipfile


from utils import load_model_config, initiate_dir


PRETRAINED_MODEL_REPO = 'https://zenodo.org/api/records/11416081'
MODEL_DIR = 'models/'


def SI_PGS(beta='0.90', device = 'cpu'):

    initiate_dir(MODEL_DIR)

    if beta == '0.90':
        key = 'SIPGS-beta0.90-pretrained.zip'
    elif beta == '0.75':
        key = 'SIPGS-beta0.75-pretrained.zip'
    elif beta == '0.50':
        key = 'SIPGS-beta0.50-pretrained.zip'
    elif beta == '0.25':
        key = 'SIPGS-beta0.25-pretrained.zip'
    else:
        print(f'beta = {beta} not a valid model')


    ##Check if model already exists

    model_location = os.path.join(MODEL_DIR, key.split(".zip")[0])

    if not os.path.exists(model_location) or not os.listdir(model_location):

        print(f'No model found, downloading {key} from {PRETRAINED_MODEL_REPO}')


        initiate_dir(model_location, True)
        response = requests.get(PRETRAINED_MODEL_REPO)

        if response.status_code == 200:
            data = response.json()

            files = data['files']

            model_file = next(file for file in files if file['key'] == key)
            model_file_url= model_file['links']['self']

            model_response = requests.get(model_file_url, stream=True)
            total_size = int(model_response.headers.get('content-length', 0))
            block_size = 1024  # 1 Kilobyte

            ## Download file

            model_filename = os.path.join(MODEL_DIR, model_file['key'])
            
            with open(model_filename, 'wb') as f, tqdm(
                desc=model_filename,
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
                file=sys.stdout
            ) as bar:

                for data in model_response.iter_content(block_size):
                    f.write(data)
                    bar.update(len(data))


            print(f'Model downloaded successfully to {model_filename}...unzipping')

            with zipfile.ZipFile(model_filename, 'r') as zip_ref:
                total_files = len(zip_ref.infolist())
                
                with tqdm(total=total_files, unit = 'file', file=sys.stdout) as bar:
                    for file in zip_ref.infolist():
                        zip_ref.extract(file, MODEL_DIR)
                        bar.update(1)
                
            print('finished unzipping')
        else:
            print(f'download unsuccessful error code{response.status_code}')



    feature_encoder_params, feature_encoder_pt_file = load_model_config(f'{model_location}/feature_encoder.json')
    generator_params, generator_pt_file = load_model_config(f'{model_location}/generator.json')
    dicriminator_params, discriminator_pt_file= load_model_config(f'{model_location}/discriminator.json')


    feature_encoder = LSTM_Feature_Encoder(**feature_encoder_params).to(device)
    generator = CVAE_Generator(**generator_params).to(device)
    discriminator = Discriminator(**dicriminator_params).to(device)

    feature_encoder.load_state_dict(torch.load(f'{MODEL_DIR}{feature_encoder_pt_file}', map_location=device))
    generator.load_state_dict(torch.load(f'{MODEL_DIR}{generator_pt_file}', map_location=device))
    discriminator.load_state_dict(torch.load(f'{MODEL_DIR}{discriminator_pt_file}', map_location=device))


    return feature_encoder, generator, discriminator


#### Model Components

######################################################################################## LSTM encoder
class LSTM_Feature_Encoder(nn.Module):
    def __init__(self, feature_size, hidden_size = 128, num_layers=4, output_size= 64):
        super(LSTM_Feature_Encoder, self).__init__()

        self.params = {
            'feature_size':feature_size,
            'hidden_size':hidden_size,
            'num_layers': num_layers,
            'output_size': output_size
        }

        self.lstm = nn.LSTM(feature_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, output_size),
            nn.Tanh()
                               )
        
    def forward(self, padded_sequences, sequence_lengths):

        packed_sequences = pack_padded_sequence(padded_sequences, sequence_lengths, batch_first = True, enforce_sorted=False)

        packed_output, (hidden, cell) = self.lstm(packed_sequences)

        embedding = self.fc(hidden[-1])

        return embedding


######################################################################################## CVAE_Generator
class CVAE_Generator(nn.Module):
    def __init__(self, img_shape, latent_size, conditional_size):
        super(CVAE_Generator, self).__init__()

        self.params = {
            'img_shape':img_shape,
            'latent_size':latent_size,
            'conditional_size': conditional_size
        }

        self.img_shape = img_shape
        self.conditional_size = conditional_size
        self.input_size = self.img_shape[0] + self.conditional_size
        self.latent_size = latent_size
        
        self.encoder = nn.Sequential(
            nn.Conv2d(self.input_size, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )

        self.hidden_shape, self.hidden_size = self._get_hidden_dimensions()

        self.mu_fc = nn.Linear(self.hidden_size, self.latent_size)
        self.logvar_fc = nn.Linear(self.hidden_size, self.latent_size)
        
        self.z_fc = nn.Linear(self.latent_size + self.conditional_size, self.hidden_size)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(self.hidden_shape[0], 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def encode(self, x, y):
        x = x.view(x.size(0), self.conditional_size, 1, 1).repeat(1, 1, self.img_shape[1], self.img_shape[2])
        x = torch.cat([y, x], dim=1)
        x = self.encoder(x)
        x = x.view(x.size(0), -1) #Flatten encoder output for final Fully cunnected liear layer

        mu, logvar = self.mu_fc(x), self.logvar_fc(x)
        
        return mu, logvar

    def reparam(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)

        z = mu + eps*std
        
        return z
        
    def decode(self, z, x):

        x = x.view(x.size(0),-1)
        x = torch.cat([z,x], dim=1)
        x = self.z_fc(x)
        x = x.view(x.size(0), self.hidden_shape[0], self.hidden_shape[1],self.hidden_shape[2])
        y_hat = self.decoder(x)

        return y_hat
        

    def forward(self, x, y):
        
        mu, logvar = self.encode(x,y)
        z = self.reparam(mu, logvar)
        y_hat = self.decode(z,x)

        return y_hat, mu, logvar

    def _get_hidden_dimensions(self):

        dummy_input = torch.randn(1, self.input_size, self.img_shape[1], self.img_shape[2])

        x = self.encoder(dummy_input)
        hidden_shape = x.size()[1:]
        
        x = x.view(x.size(0), -1) #flatten to (batch_size, flattened_dimension)

        hidden_size = x.size(1)

        return hidden_shape, hidden_size 


######################################################################################## Discriminator


class Discriminator(nn.Module):
    def __init__(self, img_shape):
        super(Discriminator, self).__init__()
        
        self.params = {
            'img_shape':img_shape
        }

        self.img_shape = img_shape
        
        self.cnn = nn.Sequential(
            nn.Conv2d(self.img_shape[0], 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )

        self.hidden_shape, self.hidden_size= self._get_hidden_dimensions()

        self.fc = nn.Sequential(
            nn.Linear(self.hidden_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid()
            # nn.Tanh()
        )

    def forward(self, y):
        x = self.cnn(y)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    def _get_hidden_dimensions(self):

        dummy_input = torch.randn(1, self.img_shape[0], self.img_shape[1], self.img_shape[2])
        x = self.cnn(dummy_input)
        hidden_shape = x.size()[1:]
        
        x = x.view(x.size(0), -1) #flatten to (batch_size, flattened_dimension)
        hidden_size = x.size(1)

        return hidden_shape, hidden_size 


if __name__ == "__main__":
    feature_encoder, generator, discriminator= SI_PGS(beta='0.75')
    
    print(feature_encoder)
    print(generator)
    print(discriminator)


