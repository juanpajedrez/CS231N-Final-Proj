import torch
import torchvision.transforms as transforms
import torchvision.models as models
from scipy.linalg import sqrtm
import numpy as np
from torch.utils.data import DataLoader

def calculate_activation_statistics(images, model, batch_size=50, dims=2048, device='cpu'):
    model.eval()
    act = np.empty((len(images), dims))

    if not isinstance(images, DataLoader):
        images = DataLoader(images, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for i, batch in enumerate(images):
            if isinstance(batch, list):
                batch = batch[0]
            batch = batch.to(device)
            pred = model(batch)[0]

            # Resize from (batch_size, dims, 1, 1) to (batch_size, dims)
            if pred.size(2) != 1 or pred.size(3) != 1:
                pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
            act[i * batch_size: i * batch_size + batch.size(0)] = pred.cpu().numpy().reshape(batch.size(0), -1)

    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma

def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance."""
    covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary parts
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    # {\displaystyle d^{2}=|\mu _{X}-\mu _{Y}|^{2}+\operatorname {tr} (\Sigma _{X}+\Sigma _{Y}-2(\Sigma _{X}\Sigma _{Y})^{1/2})}.
    mean_diff = mu1 - mu2
    return mean_diff.dot(mean_diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)

def get_pretrained_inception_v3(device='cpu'):
    model = models.inception_v3(pretrained=True)
    model.to(device)
    model.eval()
    return model

def calculate_fid(real_images, fake_images, device='cpu'):
    """ Calculates the FID score between two datasets of images. """
    inception_v3 = get_pretrained_inception_v3(device=device)
    m1, s1 = calculate_activation_statistics(real_images, inception_v3, device=device)
    m2, s2 = calculate_activation_statistics(fake_images, inception_v3, device=device)
    fid_value = frechet_distance(m1, s1, m2, s2)
    return fid_value
