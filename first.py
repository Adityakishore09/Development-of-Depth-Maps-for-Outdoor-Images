"""
This code is for training the DepthAnything model encoder
"""


import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import os
from PIL import Image
from transformers import AutoModelForDepthEstimation
import matplotlib.pyplot as plt


"""Model arguments."""  #Would be better if we take them as args input, but for now, this works

# Using HuggingFace for importing the model
depth_model = AutoModelForDepthEstimation.from_pretrained("LiheYoung/depth-anything-small-hf")
# Setting the device
device = 'cuda:0' if torch.cuda.is_available() else 'cpu' # On tmux, cuda:1 seems unavailable...
device = torch.device(device)
depth_model = depth_model.to(device)
# Output dimension of the encoder
enc_feat_size = 384
# Number of epochs after which to save
save_epochs = 5
# Directories for dataset
day_images_root = '/data2/dse411a/project3/team2/sample_images/daynight_backup/day_frames'  
night_images_root = '/data2/dse411a/project3/team2/sample_images/daynight_backup/night_frames'
# Batch size for the dataloader
batch_size = 32
# Number of epochs for the training
n_epochs = 40
# Name of the experiment. Change it for each run
expt_name = 'run3_Encoder_finetune'
# If True, the encoder is trained from scratch instead of being finetuned
train_enc = False

# Creating folder to save experiment results.
save_path = f'./{expt_name}_wts/'
try: 
    os.mkdir(save_path)
    print(f"Directory '{save_path}' created!")
except: 
    print(f"Directory '{save_path}' already exists...")

# Transformations applied to the input
transformations = transforms.Compose([transforms.Resize((224, 224)),
                                        transforms.ToTensor(),
                                        ])


"""PatchGAN Discriminator"""

# PatchGAN discriminator (modified from github/dome272/VQGAN-pytorch)
class Discriminator(nn.Module):
    def __init__(self, image_channels, num_filters_last=64, n_layers=3, kernel_size= 3):
        super(Discriminator, self).__init__()

        layers = [nn.Conv2d(image_channels, num_filters_last, kernel_size, stride=2, padding= 1), nn.LeakyReLU(0.2)]
        num_filters_mult = 1

        for i in range(1, n_layers + 1):
            num_filters_mult_last = num_filters_mult
            num_filters_mult = min(2 ** i, 8)
            layers += [
                nn.Conv2d(num_filters_last * num_filters_mult_last, num_filters_last * num_filters_mult, kernel_size,
                          2 if i < n_layers else 1, 1, bias=False),
                nn.BatchNorm2d(num_filters_last * num_filters_mult),
                nn.LeakyReLU(0.2, True)
            ]

        layers.append(nn.Conv2d(num_filters_last * num_filters_mult, 1, kernel_size, 1, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


"""Dataset preparation"""

# Function to get the file names of all image files in the given folder
def get_file_names(root):
    imgs = []
    for path, subdirs, files in os.walk(root):
        for name in files:
            imgs.append(os.path.join(path, name))
    return imgs

# Dataset class
class DayNightDataset(Dataset):
    def __init__(self, day_images, night_images, transformations):
        self.day_images = day_images
        self.night_images = night_images
        self.transformations = transformations

    def __len__(self):
        return len(self.night_images)

    def __getitem__(self, idx):
        night_img = Image.open(self.night_images[idx]).convert('RGB')
        day_img = Image.open(self.day_images[idx]).convert('RGB')

        if transformations is not None:
            night_img = transformations(night_img)
            day_img = transformations(day_img)

        return day_img, night_img


""" Model Definition """

# Taking the encoder from the backbone of the Depth-Anything model.
encoder_frozen = depth_model.backbone
encoder_train = depth_model.backbone
discriminator = Discriminator(image_channels= enc_feat_size).to(device)

# Freeze the pre-trained encoder 
for param in encoder_frozen.parameters():
    param.requires_grad = False

# Reinitialize weights for encoder_train if needed
if train_enc:
    for name, param in encoder_train.state_dict().items():
        if param.dim() > 1:
            torch.nn.init.xavier_uniform_(param)
        elif param.dim() == 1:
            torch.nn.init.zeros_(param)
        else: 
            continue


# Loading the dataset
day_images = get_file_names(day_images_root)
night_images = get_file_names(night_images_root)

max_len = max(len(day_images), len(night_images))
day_images = day_images[:max_len]
night_images = night_images[:max_len]

dataset = DayNightDataset(day_images, night_images, transformations)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last= True)


"""Defining model properties"""
learn_rate_enc = 1e-4
learn_rate_disc = 5e-5
loss_func = nn.BCEWithLogitsLoss()
optimizer_enc = optim.Adam(encoder_train.parameters(), lr=learn_rate_enc)
optimizer_disc = optim.Adam(discriminator.parameters(), lr=learn_rate_disc)


# Training the night encoder

encoder_losses = []
discriminator_losses = []

for epoch in tqdm(range(n_epochs), desc= 'Epochs', ncols= 80):
    encoder_train.train()
    discriminator.train()
    for day_img, night_img in tqdm(loader, desc= 'Batch', leave= False, ncols= 80):
        day_img, night_img = day_img.to(device), night_img.to(device)
        

        # Taking only the high-level feature map. Can try concatenating all four of them later if needed
        with torch.no_grad(): 
            day_feat = encoder_frozen(day_img).feature_maps[-1]
        
        night_feat = encoder_train(night_img).feature_maps[-1]

        # The feature maps are 3d, while the discriminator wants 4d input. So, removing the CLS token from the maps and then reshaping as per the disciminator
        day_feat = day_feat[:, 1:, :]
        night_feat = night_feat[:, 1:, :]
        size = int(day_feat.size(1)**0.5)
        day_feat = day_feat.permute(0, 2, 1).reshape(batch_size, -1, size, size)
        night_feat = night_feat.permute(0, 2, 1).reshape(batch_size, -1, size, size)

        # Train Discriminator
        optimizer_disc.zero_grad()
        real_labels = torch.ones(day_feat.size(0), 1, requires_grad=False).to(device)
        fake_labels = torch.zeros(night_feat.size(0), 1, requires_grad=False).to(device)

        disc_day = discriminator(day_feat)
        disc_day = disc_day.view(disc_day.size(0), -1)
        disc_day = torch.mean(disc_day, dim= 1, keepdim= True)
        real_loss = loss_func(disc_day, real_labels)

        disc_night = discriminator(night_feat)
        disc_night = disc_night.view(disc_night.size(0), -1)
        disc_night = torch.mean(disc_night, dim= 1, keepdim= True)
        fake_loss = loss_func(disc_night, fake_labels)
        
        disc_loss = (real_loss + fake_loss) / 2
        disc_loss.backward()
        optimizer_disc.step()

        # Train Encoder
        optimizer_enc.zero_grad()
        disc_night = discriminator(night_feat)
        disc_night = disc_night.view(disc_night.size(0), -1)
        disc_night = torch.mean(disc_night, dim= 1, keepdim= True)
        encoder_loss = loss_func(disc_night, real_labels)
        encoder_loss.backward()
        optimizer_enc.step()

    tqdm.write(f"Epoch {epoch + 1}: Encoder Loss: {encoder_loss.item()}, Discriminator Loss: {disc_loss.item()}")
    encoder_losses.append(encoder_loss.item())
    discriminator_losses.append(disc_loss.item())

    # Saving the model weights after 'save_epochs'
    enc_name = f'enc_{expt_name}_{epoch+1}_epochs.pth'    
    disc_name = f'disc_{expt_name}_{epoch+1}_epochs.pth' 
    if (epoch+1) % save_epochs == 0:   
        tqdm.write("SAVING WEIGHTS OF ENCODER...", end="")
        enc_path = os.path.join(save_path, enc_name)
        torch.save(encoder_train.state_dict(), enc_path)
        tqdm.write("SAVED!")
        tqdm.write("SAVING WEIGHTS OF DISCRIMINATOR...", end="")
        disc_path = os.path.join(save_path, disc_name)
        torch.save(discriminator.state_dict(), disc_path)
        tqdm.write("SAVED!")

# Saving the loss curves of the model
if not train_enc:  all_losses = {'encoder_ft': encoder_losses, 'discriminator_ft':discriminator_losses}
else: all_losses = {'encoder_train': encoder_losses, 'discriminator_train':discriminator_losses}
for k, losses in all_losses.items():
    plt.figure()
    plt.plot(range(1, len(losses)+1), losses)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'{k} Loss curve')
    graph_name = f'{k}_loss_curves.png'
    graph_path = os.path.join(save_path, graph_name)
    plt.savefig(graph_path)