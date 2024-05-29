import os
import pandas as pd
import datetime
import time, datetime, pytz 
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import json
import matplotlib.pyplot as plt
from scipy import linalg
import random
import cv2


str_format = '%Y%m%d%H%M%S'

def normalize_image_tensor(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """normalizes a tensor of images.
    
    Args:
        tensor (torch.Tensor): The input tensor to be normalized.
        mean (list): The mean used in normalization for each channel.
        std (list): The standard deviation used in normalization for each channel.
    
    Returns:
        torch.Tensor: The normalized tensor.
    """
    if tensor.ndim == 3:  # Single image [C, H, W]
        tensor = tensor.unsqueeze(0)  # Add batch dimension [1, C, H, W]
        unsqueezed = True
    else:
        unsqueezed = False

    mean = torch.tensor(mean).view(1, -1, 1, 1).to(tensor.device)
    std = torch.tensor(std).view(1, -1, 1, 1).to(tensor.device)

    normalized_tensor = (tensor - mean) / std

    if unsqueezed:
        normalized_tensor = normalized_tensor.squeeze(0)  # Remove batch dimension if it was added
    
    return normalized_tensor     
    

def denormalize_image_tensor(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """Denormalizes a tensor of images.
    
    Args:
        tensor (torch.Tensor): The input tensor to be denormalized.
        mean (list): The mean used in normalization for each channel.
        std (list): The standard deviation used in normalization for each channel.
    
    Returns:
        torch.Tensor: The denormalized tensor.
    """
    if tensor.ndim == 3:  # Single image [C, H, W]
        tensor = tensor.unsqueeze(0)  # Add batch dimension [1, C, H, W]
        unsqueezed = True
    else:
        unsqueezed = False

    mean = torch.tensor(mean).view(1, -1, 1, 1).to(tensor.device)
    std = torch.tensor(std).view(1, -1, 1, 1).to(tensor.device)

    denormalized_tensor = tensor * std + mean

    if unsqueezed:
        denormalized_tensor = denormalized_tensor.squeeze(0)  # Remove batch dimension if it was added
    
    return denormalized_tensor    

def load_model_config(json_file):

    with open(json_file, 'r') as config_file:
        config = json.load(config_file)
        
    return config['model_params'], config['pt_file']

def save_model_config(model_type, pt_file, model_params, json_filename):
    config = {
        "model_type":model_type,
        "pt_file":pt_file,
        "model_params": model_params
        }
    
    with open(json_filename, 'w') as json_file:
        json.dump(config, json_file, indent=4)

def initiate_dir(dir):
    if not os.path.exists(dir):
        # Create the directory
        os.makedirs(dir)
        print(f"Directory '{dir}' created.")
    else:
        print(f"Directory '{dir}' already exists.")

    return dir

def save_metrics(model_name, mse_loss, fid_score, json_filename):

    config = {
        "model_name":model_name,
        "mse_loss":mse_loss,
        "fid_score": fid_score
        }
    
    with open(json_filename, 'w') as json_file:
        json.dump(config, json_file, indent=4)

    
def tensor_toImage(tensor, custom_transforms=False):

    
    image_transform = transforms.ToPILImage() if not custom_transforms else custom_transforms

    image = image_transform(tensor)

    return image

def imshow_batch(batch_size, images, dir=None, yhat=True):
    ts = datetime.datetime.now().strftime(str_format)
    ncols = 10
    nrows = batch_size // ncols + (batch_size % ncols > 0)
    
    fig = plt.figure( num=f'batched_{ts}', figsize=(25,5*nrows), layout='tight')
    for i in np.arange(batch_size):
        ax = plt.subplot(nrows, ncols, i + 1, xticks=[], yticks=[])
        # ax.set_title(z[i])
        plt.imshow(images[i].permute(1, 2, 0) if torch.is_tensor(images) else np.transpose(images[i], (1, 2, 0)) if isinstance(images, np.ndarray) else None)
    
    if dir is not None and os.path.exists(dir):        
        plt.savefig(f'{dir}experiment_batched_{"yhat" if yhat else "y"}.png')  # Save the figure to the specified directory
        # plt.close(fig)  # Close the figure to free up memor
        for idx, img in enumerate(images):
            img = np.transpose(img, (1, 2, 0))
            img = Image.fromarray((img*255).astype(np.uint8))
            img.save(f'{dir}{idx}_{"yhat" if yhat else "y"}.jpg')

    plt.show()


def calculate_fid(real_feats, gen_feats):
    # Convert list of arrays to a single numpy array
    real_feats = np.concatenate(real_feats, axis=0)
    gen_feats = np.concatenate(gen_feats, axis=0)

    mu1, sigma1 = real_feats.mean(axis=0), np.cov(real_feats, rowvar=False)
    mu2, sigma2 = gen_feats.mean(axis=0), np.cov(gen_feats, rowvar=False)

    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    covmean = linalg.sqrtm(sigma1.dot(sigma2), disp=False)[0]

    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid



def images_to_video(image_folder, video_name, fps):

    print(f'combining images to video at: {video_name}')

    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
    images.sort(key=lambda x: float(x.split('.jpg')[0]))  # sort by ascending numerical

    # Read the first image to get the width and height
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    h, w, layers = frame.shape
    size = (w, h)

    # Define the codec using VideoWriter_fourcc and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_name, fourcc, fps, size)

    for image in images:
        video_frame = cv2.imread(os.path.join(image_folder, image))
        out.write(video_frame)

    out.release()
    cv2.destroyAllWindows()
    print(f"Video {video_name} created successfully!")



# class GardynDataset(Dataset):
#     def __init__(self, annotations_file, img_dir, camera = 'camera2', cumulative = True, params = ['Temperature_C', 'Humidity_percent', 'EC', 'PH', 'WaterTemp_C'] ,  transform=None):
#         """
#         Args:
#             annotations_file (string): Path to the csv file with annotations.
#             img_dir (string): Directory with all the images.
#             transform (callable, optional): Optional transform to be applied on a sample.
#         """

#         self.img_dir = img_dir
#         self.annotations_file = annotations_file
#         self.cumulative = cumulative
#         self.transform = transform
#         self.key = params + ['timestamp', 'time_interval']
        
        
#         ## import data from data csv and filter data to entries of existing files in image dataset
#         annotation_data = pd.read_csv(self.annotations_file)
#         annotation_data['file_exists'] = annotation_data[camera].apply(lambda x: os.path.exists(os.path.join(self.img_dir, x)))
#         self.annotation_data = annotation_data
#         self.existing_data = self.annotation_data[self.annotation_data['file_exists']]

#         ## extract time stamps and image labels
#         self.img_list = self.existing_data['camera2'].tolist()
#         timestamps = self.existing_data['_time'].tolist()

#         ## Normalize timestamps and generate intervals
#         self.timestamps = [datetime.datetime.fromisoformat(ts).timestamp() for ts in timestamps]
#         self.timestamps_normalized = [ts - self.timestamps[0] for ts in self.timestamps]
#         time_intervals = [ts - self.timestamps_normalized[i] for i,ts in enumerate(self.timestamps_normalized[1:])]
#         self.time_intervals = [0] + time_intervals

#         labels = self.existing_data[params].interpolate(method='linear')
#         labels['timestamp'] = self.timestamps_normalized
#         labels['time_interval'] = self.time_intervals
#         self.labels = labels

#         self.total_sequence_length = len(self.img_list)


#     def __len__(self):
#         return len(self.img_list)

#     def __getitem__(self, idx):
#         img_path = os.path.join(self.img_dir, self.img_list[idx])
        
#         image = Image.open(img_path).convert('RGB')

#         ts = self.labels['timestamp'].iloc[idx]
        
#         if self.transform:
#             image = self.transform(image)

#         if not self.cumulative:
#             label = self.labels.iloc[idx].to_numpy()
            
#             return image, torch.from_numpy(label), torch.tensor(ts)
                      
#         else:
#             label = self.labels.iloc[:idx+1].to_numpy()

#             sequence_length = label.shape[0]

#             pad_width = ((0,self.total_sequence_length- sequence_length),(0,0))
            
#             label = np.pad(label, pad_width, 'constant', constant_values=(0))
                    
#             return image, torch.from_numpy(label).float(), torch.tensor(sequence_length), torch.tensor(ts).float()



class GardynDataset_cum(Dataset):
    def __init__(self, annotations_file, img_dir, camera = 'camera2', cumulative = True, params = ['Temperature_C', 'Humidity_percent', 'EC', 'PH', 'WaterTemp_C'] ,  transform=None):
        """
        This is the modified dataset that will return the indexed item plus 2 items in the future
        Args:
            annotations_file (string): Path to the csv file with annotations.
            img_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """

        self.img_dir = img_dir
        self.annotations_file = annotations_file
        self.cumulative = cumulative
        self.transform = transform
        self.key = params + ['timestamp', 'time_interval']
        
        
        ## import data from data csv and filter data to entries of existing files in image dataset
        annotation_data = pd.read_csv(self.annotations_file)
        annotation_data['file_exists'] = annotation_data[camera].apply(lambda x: os.path.exists(os.path.join(self.img_dir, x)))
        self.annotation_data = annotation_data
        self.existing_data = self.annotation_data[self.annotation_data['file_exists']]

        ## extract time stamps and image labels
        self.img_list = self.existing_data['camera2'].tolist()
        timestamps = self.existing_data['_time'].tolist()

        ## Normalize timestamps and generate intervals
        self.timestamps = [datetime.datetime.fromisoformat(ts).timestamp() for ts in timestamps]
        self.timestamps_normalized = [ts - self.timestamps[0] for ts in self.timestamps]
        time_intervals = [ts - self.timestamps_normalized[i] for i,ts in enumerate(self.timestamps_normalized[1:])]
        self.time_intervals = [0] + time_intervals

        labels = self.existing_data[params].interpolate(method='linear')
        labels['timestamp'] = self.timestamps_normalized
        labels['time_interval'] = self.time_intervals
        self.labels = labels

        self.total_sequence_length = len(self.img_list)


    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        
        # image = Image.open(img_path).convert('RGB')


        if not self.cumulative:
            img_path = os.path.join(self.img_dir, self.img_list[idx])
            
            image = Image.open(img_path).convert('RGB')

            if self.transform:
                image = self.transform(image)
            
            ts = self.labels['timestamp'].iloc[idx]
            
            label = self.labels.iloc[idx].to_numpy()
            
            return image, torch.from_numpy(label), torch.tensor(ts)
                      
        else:
            
            if idx<= len(self.img_list)-3:
                
                img_paths = [os.path.join(self.img_dir, self.img_list[i]) for i in range(idx, idx+3)]
                
                image = [Image.open(img_path).convert('RGB') for img_path in img_paths]
    
                if self.transform:
                    image = [self.transform(i) for i in image]
                    image = torch.stack(image,dim=0)
                
                ts = [self.labels['timestamp'].iloc[i] for i in range(idx, idx+3)]
            
                labels = [self.labels.iloc[:i+1].to_numpy() for i in range(idx, idx+3)]### PLEASE DONT FORGET THIS ASPECT  

            else:
                
                img_paths = [os.path.join(self.img_dir, self.img_list[i]) for i in range(-3, 0)]
                
                image = [Image.open(img_path).convert('RGB') for img_path in img_paths]
    
                if self.transform:
                    image = [self.transform(i) for i in image]
                    image = torch.stack(image,dim=0)
                
                ts = [self.labels['timestamp'].iloc[i] for i in range(-3, 0)]
            
                labels = [self.labels.iloc[:i].to_numpy() for i in range(-3, 0)]### PLEASE DONT FORGET THIS ASPECT  
                

            sequence_lengths = [label.shape[0] for label in labels]

            pad_widths = [((0,self.total_sequence_length- sequence_length),(0,0)) for sequence_length in sequence_lengths]
            
            labels = np.stack([np.pad(label, pad_width, 'constant', constant_values=(0)) for pad_width,label in zip(pad_widths, labels)])
            sequence_lengths = np.stack(sequence_lengths)
            
                    
            return image, torch.from_numpy(labels).float(), torch.tensor(sequence_lengths), torch.tensor(ts).float()



class GardynDataset_with_priors(Dataset):
    def __init__(self, annotations_file, img_dir, camera = 'camera2', normalized =True, cumulative = True, params = ['Temperature_C', 'Humidity_percent', 'EC', 'PH', 'WaterTemp_C'] ,  transform=None):
        """
        Args:
            annotations_file (string): Path to the csv file with annotations.
            img_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """

        self.img_dir = img_dir
        self.annotations_file = annotations_file
        self.cumulative = cumulative
        self.transform = transform
        self.key = params + ['timestamp', 'time_interval']
        
        
        ## import data from data csv and filter data to entries of existing files in image dataset
        annotation_data = pd.read_csv(self.annotations_file)
        annotation_data['file_exists'] = annotation_data[camera].apply(lambda x: os.path.exists(os.path.join(self.img_dir, x)))
        self.annotation_data = annotation_data
        self.existing_data = self.annotation_data[self.annotation_data['file_exists']]

        ## extract time stamps and image labels
        self.img_list = self.existing_data['camera2'].tolist()
        timestamps = self.existing_data['_time'].tolist()

        ## Normalize timestamps and generate intervals
        self.timestamps = [datetime.datetime.fromisoformat(ts).timestamp() for ts in timestamps]
        self.timestamps = [ts - self.timestamps[0] for ts in self.timestamps]
        timestamps_np = np.array(self.timestamps)
        min_time = np.min(timestamps_np)
        max_time = np.max(timestamps_np)

        normalized_timestamps_np = (timestamps_np - min_time) / (max_time - min_time)
        self.timestamps_normalized = normalized_timestamps_np.tolist()

        

        labels = self.existing_data[params].interpolate(method='linear')

        if normalized:
            for param in params:
                labels[param] = (labels[param] - labels[param].min()) / (labels[param].max() - labels[param].min())

            time_intervals = [ts - self.timestamps_normalized[i] for i,ts in enumerate(self.timestamps_normalized[1:])]
            labels['timestamp'] = self.timestamps_normalized

        else:
            time_intervals = [ts - self.timestamps[i] for i,ts in enumerate(self.timestamps[1:])]
            labels['timestamp'] = self.timestamps

        self.time_intervals = [0] + time_intervals
        
        labels['time_interval'] = self.time_intervals
        self.labels = labels

        self.total_sequence_length = len(self.img_list)


    def __len__(self):
        return len(self.img_list)

    def get_item(self,idx):
        
        img_path = os.path.join(self.img_dir, self.img_list[idx])
        
        image = Image.open(img_path).convert('RGB')

        ts = self.labels['timestamp'].iloc[idx]
        
        if self.transform:
            image = self.transform(image)

        if not self.cumulative:
            label = self.labels.iloc[idx].to_numpy()
            
            return image, torch.from_numpy(label), torch.tensor(ts)
                      
        else:
            label = self.labels.iloc[:idx+1].to_numpy()
            
            sequence_length = label.shape[0]

            pad_width = ((0,self.total_sequence_length- sequence_length),(0,0))
            
            label = np.pad(label, pad_width, 'constant', constant_values=(0))
                    
            return image, torch.from_numpy(label).float(), torch.tensor(sequence_length), torch.tensor(ts).float()


    def __getitem__(self, idx):

        if idx < 1:
            image_prior, label_prior, sequence_length_prior, timestamp_prior = self.get_item(idx)
            image, label, sequence_length, timestamp = self.get_item(random.randint(idx+1, self.__len__()))

        else:
            image, label, sequence_length, timestamp = self.get_item(idx)
            image_prior, label_prior, sequence_length_prior, timestamp_prior = self.get_item(random.randint(0, idx-1))

        return image, label, sequence_length, timestamp, image_prior, label_prior, sequence_length_prior, timestamp_prior 
        
        



class GardynDataset(Dataset):
    def __init__(self, annotations_file, img_dir, camera = 'camera2', normalized =True, cumulative = True, params = ['Temperature_C', 'Humidity_percent', 'EC', 'PH', 'WaterTemp_C'] ,  transform=None):
        """
        Args:
            annotations_file (string): Path to the csv file with annotations.
            img_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """

        self.img_dir = img_dir
        self.annotations_file = annotations_file
        self.cumulative = cumulative
        self.transform = transform
        self.key = params + ['timestamp', 'time_interval']
        
        
        ## import data from data csv and filter data to entries of existing files in image dataset
        annotation_data = pd.read_csv(self.annotations_file)
        annotation_data['file_exists'] = annotation_data[camera].apply(lambda x: os.path.exists(os.path.join(self.img_dir, x)))
        self.annotation_data = annotation_data
        self.existing_data = self.annotation_data[self.annotation_data['file_exists']]

        ## extract time stamps and image labels
        self.img_list = self.existing_data['camera2'].tolist()
        timestamps = self.existing_data['_time'].tolist()

        ## Normalize timestamps and generate intervals
        self.timestamps = [datetime.datetime.fromisoformat(ts).timestamp() for ts in timestamps]
        self.timestamps = [ts - self.timestamps[0] for ts in self.timestamps]
        timestamps_np = np.array(self.timestamps)
        min_time = np.min(timestamps_np)
        max_time = np.max(timestamps_np)

        normalized_timestamps_np = (timestamps_np - min_time) / (max_time - min_time)
        self.timestamps_normalized = normalized_timestamps_np.tolist()

        

        labels = self.existing_data[params].interpolate(method='linear')

        if normalized:
            for param in params:
                labels[param] = (labels[param] - labels[param].min()) / (labels[param].max() - labels[param].min())

            time_intervals = [ts - self.timestamps_normalized[i] for i,ts in enumerate(self.timestamps_normalized[1:])]
            labels['timestamp'] = self.timestamps_normalized

        else:
            time_intervals = [ts - self.timestamps[i] for i,ts in enumerate(self.timestamps[1:])]
            labels['timestamp'] = self.timestamps

        self.time_intervals = [0] + time_intervals
        
        labels['time_interval'] = self.time_intervals
        self.labels = labels

        self.total_sequence_length = len(self.img_list)


    def __len__(self):
        return len(self.img_list)

    def __getitem__(self,idx):
        
        img_path = os.path.join(self.img_dir, self.img_list[idx])
        
        image = Image.open(img_path).convert('RGB')

        ts = self.labels['timestamp'].iloc[idx]
        
        if self.transform:
            image = self.transform(image)

        if not self.cumulative:
            label = self.labels.iloc[idx].to_numpy()
            
            return image, torch.from_numpy(label).float(), torch.tensor(ts).float()
                      
        else:
            label = self.labels.iloc[:idx+1].to_numpy()
            
            sequence_length = label.shape[0]

            pad_width = ((0,self.total_sequence_length- sequence_length),(0,0))
            
            label = np.pad(label, pad_width, 'constant', constant_values=(0))
                    
            return image, torch.from_numpy(label).float(), torch.tensor(sequence_length), torch.tensor(ts).float()







