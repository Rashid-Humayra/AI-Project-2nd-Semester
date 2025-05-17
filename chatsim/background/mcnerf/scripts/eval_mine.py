import numpy as np
import lpips
import click
import os
import json
import torch
import scipy
import scipy.signal
from tqdm import tqdm
from glob import glob
from os.path import join as pjoin
from skimage import io as imageio
from skimage.metrics import peak_signal_noise_ratio
import wandb  # ✅ added wandb import


def glob_images(image_dir):
    ret = []
    for suff in ['*.jpg', '*.JPG', '*.png', '*.PNG']:
        ret += glob(pjoin(image_dir, suff))
    return sorted(ret)


def to_torch_image(img):
    return ((torch.from_numpy(img).to(torch.float32) / 255.0) * 2. - 1.).to(torch.device('cuda')).permute(2, 0, 1)[None]


''' Evaluation metrics (ssim, lpips)
'''
def rgb_ssim(img0, img1, max_val,
             filter_size=11,
             filter_sigma=1.5,
             k1=0.01,
             k2=0.03,
             return_map=False):
    assert len(img0.shape) == 3
    assert img0.shape[-1] == 3
    assert img0.shape == img1.shape

    hw = filter_size // 2
    shift = (2 * hw - filter_size + 1) / 2
    f_i = ((np.arange(filter_size) - hw + shift) / filter_sigma)**2
    filt = np.exp(-0.5 * f_i)
    filt /= np.sum(filt)

    def convolve2d(z, f):
        return scipy.signal.convolve2d(z, f, mode='valid')

    filt_fn = lambda z: np.stack([
        convolve2d(convolve2d(z[...,i], filt[:, None]), filt[None, :])
        for i in range(z.shape[-1])], -1)
    mu0 = filt_fn(img0)
    mu1 = filt_fn(img1)
    mu00 = mu0 * mu0
    mu11 = mu1 * mu1
    mu01 = mu0 * mu1
    sigma00 = filt_fn(img0**2) - mu00
    sigma11 = filt_fn(img1**2) - mu11
    sigma01 = filt_fn(img0 * img1) - mu01

    sigma00 = np.maximum(0., sigma00)
    sigma11 = np.maximum(0., sigma11)
    sigma01 = np.sign(sigma01) * np.minimum(
        np.sqrt(sigma00 * sigma11), np.abs(sigma01))
    c1 = (k1 * max_val)**2
    c2 = (k2 * max_val)**2
    numer = (2 * mu01 + c1) * (2 * sigma01 + c2)
    denom = (mu00 + mu11 + c1) * (sigma00 + sigma11 + c2)
    ssim_map = numer / denom
    ssim = np.mean(ssim_map)
    return ssim_map if return_map else ssim


@click.command()
@click.option('--base_data_dir', type=str, default='/home/ppwang/Projects/SANR/exp/evals')
@click.option('--scenes', type=str)
@click.option('--methods', type=str)
def main(base_data_dir, scenes, methods):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)
    scenes = scenes.split(',')
    methods = methods.split(',')

    for scene in scenes:
        scene_dir = pjoin(base_data_dir, scene)
        gt_image_paths = glob_images(pjoin(scene_dir, 'images'))

        for method in methods:
            # ✅ Initialize wandb for each (scene, method)
            wandb.init(
                project="my_eval_project",
                name=f"eval_{scene}_{method}",
                config={
                    "scene": scene,
                    "method": method,
                    "base_data_dir": base_data_dir
                }
            )

            pd_image_paths = glob_images(pjoin(base_data_dir, method))
            psnr_tot, ssim_tot, lpips_tot = 0., 0., 0.
            info_data = {'psnr': dict(), 'ssim': dict(), 'lpips': dict()}
            assert len(gt_image_paths) == len(pd_image_paths), f"GT and predicted image counts mismatch: {len(gt_image_paths)} vs {len(pd_image_paths)}"

            for i, (gt_path, pd_path) in tqdm(enumerate(zip(gt_image_paths, pd_image_paths))):
                gt_image = imageio.imread(gt_path)[:, :, :3]
                pd_image = imageio.imread(pd_path)[:, :, :3]

                psnr = peak_signal_noise_ratio(gt_image, pd_image)
                ssim = rgb_ssim(gt_image / 255., pd_image / 255., max_val=1)
                lpip = loss_fn_vgg(to_torch_image(gt_image).to(device), to_torch_image(pd_image).to(device)).cpu().item()

                psnr_tot += psnr
                ssim_tot += ssim
                lpips_tot += lpip
                info_data['psnr'][str(i)] = psnr
                info_data['ssim'][str(i)] = ssim
                info_data['lpips'][str(i)] = lpip

                # ✅ log per-image metrics
                wandb.log({
                    "image_index": i,
                    "psnr": psnr,
                    "ssim": ssim,
                    "lpips": lpip
                })

            n_images = len(gt_image_paths)
            info_data['psnr']['mean'] = psnr_tot / n_images
            info_data['ssim']['mean'] = ssim_tot / n_images
            info_data['lpips']['mean'] = lpips_tot / n_images

            # ✅ log mean metrics
            wandb.log({
                "psnr_mean": info_data['psnr']['mean'],
                "ssim_mean": info_data['ssim']['mean'],
                "lpips_mean": info_data['lpips']['mean']
            })

            # save json
            with open(pjoin(scene_dir, method, 'info.json'), 'w') as f:
                json.dump(info_data, f, indent=2)

            # ✅ finish wandb run
            wandb.finish()


if __name__ == '__main__':
    main()
