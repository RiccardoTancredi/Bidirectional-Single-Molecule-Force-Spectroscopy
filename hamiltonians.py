import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import os
plt.rcParams.update({'font.size': 12.5})

saved_imgs_folder = 'imgs/gif/'

def H0(z):
    return 5*z**4 - 10*z**2 +3*z

def Vt(z, z_t):
    return 15/2*(z-z_t)**2

def H(z, z_t):
    return H0(z) + Vt(z, z_t)

z_t = np.array([-1.5] + np.linspace(-1.5, 1.5, 750).tolist() + [1.5])
zz = np.linspace(-1.5, 1.5, 500)

if len(os.listdir(saved_imgs_folder)) == 0:
    for tt in tqdm(range(len(z_t))):
        plt.plot(zz, list(map(H0, zz)), label=r'$H_0$', 
                color='royalblue', ls=':')
        plt.plot(zz, Vt(zz, z_t[tt]), 
            label=r'$V_t$', color='darkorange', ls='--')
        plt.plot(zz, H(zz, z_t[tt]), label=r'$H_t$', color='limegreen')
        
        plt.ylim(-10, 15)
        plt.xlabel(r'$x$')
        plt.ylabel(r'$H(x)$')
        plt.grid(ls='--', alpha=0.5)
        plt.legend(loc='lower right')
        plt.savefig(f'imgs/gif/Hamiltonian{tt}.png', dpi=150)
        
        plt.close()


def make_gif(frame_folder, name='imgs/Hamiltonian.gif', jumps=8):
    tot_range = len(os.listdir(frame_folder))
    key_name, ext = os.listdir(frame_folder)[0].split('0')
    images = [f'{frame_folder}{key_name}{ii}{ext}' for ii in range(tot_range)]
    frames = [Image.open(image) for image in images][::jumps]
    frames = frames + frames[::-1]
    frame_one = frames[0]
    frame_one.save(name, format="GIF", append_images=frames,
                   save_all=True, duration=1e-3, loop=0)


make_gif(saved_imgs_folder)
for image in os.listdir(saved_imgs_folder):
    os.remove(f'{saved_imgs_folder}{image}')