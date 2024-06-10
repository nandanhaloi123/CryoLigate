from dataloader import EMdata
from torch.utils.data import DataLoader
from unet import UNet3D
import time
from diffusion import *
from torch.optim import Adam
import torch.nn.functional as F
import torch
import joblib
import mrcfile

@torch.no_grad()
def sample_timestep(x, t):
    """
    Calls the model to predict the noise in the image and returns 
    the denoised image. 
    Applies noise to this image, if we are not in the last step yet.
    """
    betas_t = get_index_from_list(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = get_index_from_list(sqrt_recip_alphas, t, x.shape)
    
    # Call model (current image - noise prediction)
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * trained_model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )
    posterior_variance_t = get_index_from_list(posterior_variance, t, x.shape)
    
    if t == 0:
        # As pointed out by Luis Pereira (see YouTube comment)
        # The t's are offset from the t's in the paper
        return model_mean
    else:
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise 
    
    
    
trained_model = joblib.load('model_saved.pkl')

img = torch.randn(1, 1, 64, 64, 64)

T = 5
for i in range(0,T)[::-1]:
    t = torch.full((1,), i, dtype=torch.long)
    img = sample_timestep(img, t)
    mrcfile.write(f'../dataset/sampled_{i}.mrc', img[0][0].numpy(), overwrite=True)
