import torch
import torch.nn as nn
from torch.utils.data import random_split
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from tqdm import tqdm
import sys

import requests
import os
import zipfile

from utils import load_model_config, initiate_dir


PRETRAINED_MODEL_REPO = 'https://zenodo.org/api/records/11495687'
MODEL_DIR = 'models/'


def download_model(key):
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
        sys.exit()

def SI_PGS(beta='0.90', device = 'cpu', custom_dir = False):

    if custom_dir:
        model_location = custom_dir
    else:
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
    
            download_model(key)

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


def SI_PGS_R(device = 'cpu', custom_dir = False):

    if custom_dir:
        model_location = custom_dir
    else:
        initiate_dir(MODEL_DIR)
    
        key = 'SIPGS-R-pretrained.zip'
    
        ##Check if model already exists
    
        model_location = os.path.join(MODEL_DIR, key.split(".zip")[0])
    
        if not os.path.exists(model_location) or not os.listdir(model_location):
    
            print(f'No model found, downloading {key} from {PRETRAINED_MODEL_REPO}')
    
            initiate_dir(model_location, True)
    
            download_model(key)

    feature_encoder_params, feature_encoder_pt_file = load_model_config(f'{model_location}/feature_encoder.json')
    generator_params, generator_pt_file = load_model_config(f'{model_location}/generator.json')
    dicriminator_params, discriminator_pt_file= load_model_config(f'{model_location}/discriminator.json')


    feature_encoder = LSTM_Feature_Encoder(**feature_encoder_params).to(device)
    generator = Recurrent_CVAE_Generator(**generator_params).to(device)
    discriminator = Recurrent_Discriminator(**dicriminator_params).to(device)

    feature_encoder.load_state_dict(torch.load(f'{MODEL_DIR}{feature_encoder_pt_file}', map_location=device))
    generator.load_state_dict(torch.load(f'{MODEL_DIR}{generator_pt_file}', map_location=device))
    discriminator.load_state_dict(torch.load(f'{MODEL_DIR}{discriminator_pt_file}', map_location=device))

    return feature_encoder, generator, discriminator



def cGAN(device = 'cpu', custom_dir = False):
    if custom_dir:
        model_location = custom_dir
    else:
        initiate_dir(MODEL_DIR)
    
        key = 'cGAN-pretrained.zip'
    
        ##Check if model already exists
    
        model_location = os.path.join(MODEL_DIR, key.split(".zip")[0])
    
        if not os.path.exists(model_location) or not os.listdir(model_location):
    
            print(f'No model found, downloading {key} from {PRETRAINED_MODEL_REPO}')
    
            initiate_dir(model_location, True)
    
            download_model(key)

    generator_params, generator_pt_file = load_model_config(f'{model_location}/generator.json')
    dicriminator_params, discriminator_pt_file= load_model_config(f'{model_location}/discriminator.json')

    generator = CGAN_Generator(**generator_params).to(device)
    discriminator = CGAN_Discriminator(**dicriminator_params).to(device)

    generator.load_state_dict(torch.load(f'{MODEL_DIR}{generator_pt_file}', map_location=device))
    discriminator.load_state_dict(torch.load(f'{MODEL_DIR}{discriminator_pt_file}', map_location=device))

    return generator, discriminator


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


######################################################################################## Recurrent CVAE Generator
class Recurrent_CVAE_Generator(nn.Module):
    def __init__(self, img_shape, latent_size, conditional_size):
        super(Recurrent_CVAE_Generator, self).__init__()

        self.params = {
            'img_shape':img_shape,
            'latent_size':latent_size,
            'conditional_size': conditional_size
        }

        self.img_shape = img_shape
        self.conditional_size = conditional_size +1  #+1 accounts for the time difference vector
        self.input_size = (2*self.img_shape[0]) + self.conditional_size 
        self.latent_size = latent_size
        self.sampling_size = self.latent_size + self.conditional_size + self.img_shape[0]
        
        self.encoder = nn.Sequential(
            nn.Conv2d(self.input_size, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )
        
        self.z_cnn = nn.Sequential(
            nn.Conv2d(self.sampling_size, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )


        self.hidden_shape, self.hidden_size, self.sample_hidden_shape, self.sample_hidden_size = self._get_hidden_dimensions()

        self.mu_fc = nn.Linear(self.hidden_size, self.latent_size)
        self.logvar_fc = nn.Linear(self.hidden_size, self.latent_size)

        self.z_fc = nn.Linear(self.sample_hidden_size, self.hidden_size)
        
        
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

    def encode(self, x, y, y_prior, time_diff):

        x = x.view(x.size(0),-1)

        x = torch.cat([x,time_diff],dim=1)
        
        x = x.view(x.size(0), self.conditional_size, 1, 1).repeat(1, 1, self.img_shape[1], self.img_shape[2])

        x = torch.cat([y, x, y_prior], dim=1)

        x = self.encoder(x)

        x = x.view(x.size(0), -1) #Flatten encoder output for final Fully cunnected liear layer

        mu, logvar = self.mu_fc(x), self.logvar_fc(x)

        return mu, logvar

    def reparam(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)

        z = mu + eps*std
        
        return z
        
    def decode(self, z, x, y_prior, time_diff):
        
        x = x.view(x.size(0),-1)

        x = torch.cat([z,x,time_diff], dim=1)

        x = x.view(x.size(0), x.size(1), 1, 1).repeat(1, 1, self.img_shape[1], self.img_shape[2])
        
        x = torch.cat([x,y_prior], dim=1)

        x = self.z_cnn(x)

        x = x.view(x.size(0),-1)
        
        x = self.z_fc(x)

        x = x.view(x.size(0), self.hidden_shape[0], self.hidden_shape[1],self.hidden_shape[2])

        y_hat = self.decoder(x)

        return y_hat
        

    def forward(self, x, y, y_prior, time_diff):

        mu, logvar = self.encode(x,y, y_prior, time_diff)
        z = self.reparam(mu, logvar)        
        y_hat = self.decode(z,x,y_prior,time_diff)
        
        return y_hat, mu, logvar

    def _get_hidden_dimensions(self):

        dummy_input = torch.randn(1, self.input_size, self.img_shape[1], self.img_shape[2])
        dummy_sampling_input = torch.randn(1, self.sampling_size, self.img_shape[1], self.img_shape[2])

        x = self.encoder(dummy_input)
        hidden_shape = x.size()[1:]
        
        x = x.view(x.size(0), -1) #flatten to (batch_size, flattened_dimension)

        hidden_size = x.size(1)

        z = self.z_cnn(dummy_sampling_input)
        
        sample_hidden_shape = z.size()[1:]
        
        z = z.view(z.size(0), -1)
        
        sample_hidden_size = z.size(1)

        return hidden_shape, hidden_size , sample_hidden_shape, sample_hidden_size

######################################################################################## Recurrent CVAE Discriminator
class Recurrent_Discriminator(nn.Module):
    def __init__(self, img_shape):
        super(Recurrent_Discriminator, self).__init__()
        
        self.params = {
            'img_shape':img_shape
        }

        self.img_shape = img_shape
        
        self.cnn = nn.Sequential(
            nn.Conv2d(self.img_shape[0], 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )

        self.hidden_shape, self.hidden_size= self._get_hidden_dimensions()

        self.fc = nn.Sequential(
            nn.Linear(self.hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
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


######################################################################################## CGAN Generator

class CGAN_Generator(nn.Module):
    def __init__(self, img_shape, conditional_size, noise_dim):
        super(CGAN_Generator, self).__init__()
        self.params = {
            'img_shape':img_shape,
            'conditional_size': conditional_size,
            'noise_dim': noise_dim
        }

        self.img_shape = img_shape
        self.img_size = img_shape[1]*img_shape[2]
        self.conditional_size = conditional_size
        self.noise_dim = noise_dim
        self.hidden_shape = (3,15,20)
        self.hidden_size = self.hidden_shape[0]*self.hidden_shape[1]*self.hidden_shape[2]

        self.input_size = self.conditional_size + noise_dim

        self.reshape_fc = nn.Linear(self.input_size, self.hidden_size)

        self.model = nn.Sequential(


            nn.ConvTranspose2d(self.hidden_shape[0], 512, kernel_size=4, stride=2, padding=1),  # (30x40)
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # (60x80)
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # (120x160)
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # (240x320)
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(64, self.img_shape[0], kernel_size=4, stride=2, padding=1), #(480x640)
            nn.Sigmoid()
            # nn.Tanh()
        )

    

    def forward(self, noise, labels):

        x = torch.cat([noise, labels], 1)

        x = self.reshape_fc(x)

        x = x.view(x.size(0),self.hidden_shape[0],self.hidden_shape[1], self.hidden_shape[2] )
        
        return self.model(x)


######################################################################################## CGAN Discriminator

class CGAN_Discriminator(nn.Module):
    def __init__(self, img_shape, conditional_size):
        super(CGAN_Discriminator, self).__init__()
        self.params = {
            'img_shape':img_shape,
            'conditional_size': conditional_size
        }

        self.img_shape = img_shape
        self.conditional_size = conditional_size
        self.input_size = self.img_shape[0] + self.conditional_size
        
        self.model = nn.Sequential(
            nn.Conv2d(self.input_size, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # (120x160)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # (60x80)
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),

            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # (30x40)
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),

            nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1),  # (15x20)
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            
            nn.Conv2d(1024, 1, kernel_size=4, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        labels = labels.unsqueeze(2).unsqueeze(3)
        labels = labels.repeat(1, 1, img.size(2), img.size(3))
        
        x = torch.cat([img, labels], 1)
        return self.model(x)



if __name__ == "__main__":
    feature_encoder, generator, discriminator= SI_PGS(beta='0.75')
    R_feature_encoder, R_generator, R_discriminator= SI_PGS_R()
    cgan_generator, cgan_discriminator= cGAN()

    print(feature_encoder)
    print(generator)
    print(discriminator)

    print(R_feature_encoder)
    print(R_generator)
    print(R_discriminator)

    print(cgan_generator)
    print(cgan_discriminator)


