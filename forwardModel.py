import numpy as np
import torch
from torch.distributions import Poisson, Normal
import matplotlib.pyplot as plt
from canopyPlots import createCHM

# set random seeds
torch.manual_seed(0)
np.random.seed(0)

def forward_model(high_res_cube, gaussian_noise_std=0.12, photon_count=20):
    high_res_cube = high_res_cube / (high_res_cube.sum(0) + 1e-8)
    distribution = Poisson(photon_count * high_res_cube)
    high_res_cube = distribution.sample()
    if gaussian_noise_std > 0:
        distribution = Normal(high_res_cube, gaussian_noise_std)
        high_res_cube = distribution.sample()

    return high_res_cube



if __name__ == "__main__":
    cube = torch.from_numpy(np.load('TestCube/gt3.npy')).float()
    
    location = (0,100)
    chm_high_res,dtm_high_res,hillshade_high_res,_ = createCHM(cube.numpy())
    cube_low = forward_model(cube)
    chm_low_res,dtm_low_res,hillshade_low_res,_ = createCHM(cube_low.numpy())

    fig, ax = plt.subplots(2, 4, figsize=(16, 8))

    ax[0,0].imshow(chm_high_res)
    ax[0,1].imshow(dtm_high_res,cmap='copper')
    ax[0,1].imshow(hillshade_high_res,cmap='Grays',alpha=0.35)
    ax[0,2].imshow(cube.numpy()[::-1,-1,:], cmap='gray_r')
    ax[0,3].plot(cube[:,location[0],location[1]].numpy())

    ax[1,0].imshow(chm_low_res)
    ax[1,1].imshow(dtm_low_res,cmap='copper')
    ax[1,1].imshow(hillshade_low_res,cmap='Grays',alpha=0.35)
    ax[1,2].imshow(cube_low.numpy()[::-1,-1,:]/(cube_low.numpy()[::-1,0,:].max(0)+1e-8), cmap='gray_r')
    ax[1,3].plot(cube_low[:,location[0],location[1]].numpy())

    ax[0,0].set_title('Canopy Height Model')
    ax[0,1].set_title('Digital Terrain Model')
    ax[0,2].set_title('Canopy Height Profile')
    ax[0,3].set_title('Single Location Waveform')

    plt.show()


    print(cube.shape, cube.min(), cube.max())