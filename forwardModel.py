import numpy as np
import torch
from torch.distributions import Poisson, Normal
import matplotlib.pyplot as plt

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
    fig, ax = plt.subplots(1, 2)

    location = (0,100)

    ax[0].plot(cube[:,location[0],location[1]].numpy())
    ax[0].set_title("Original Cube")

    cube = forward_model(cube)
    ax[1].plot(cube[:,location[0],location[1]].numpy())
    ax[1].set_title("After Forward Model")
    plt.show()


    print(cube.shape, cube.min(), cube.max())