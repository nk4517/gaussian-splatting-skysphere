import math

import torch


def gen_gaussian_ellipse_torch(img_w, img_h, sigma=1.):
    sqrt_sigma = math.sqrt(sigma * img_w)

    ax1 = torch.linspace(-img_h / 2, img_h / 2 - 1, img_h)
    gauss1 = torch.exp(-0.5 * torch.square(ax1) / sqrt_sigma)

    ax2 = torch.linspace(-img_w / 2, img_w / 2 - 1, img_w)
    gauss2 = torch.exp(-0.5 * torch.square(ax2) / sqrt_sigma)

    kernel = torch.ger(gauss1, gauss2)
    kernel /= torch.max(kernel)
    return kernel


def calc_phase_dist(fft_gt, fft_rendered):
    # phi_gt = torch.angle(fft_gt)
    # phi = torch.angle(fft_rendered)
    # d_phi = phi_gt - phi

    # phi_dist = torch.atan2(torch.sin(d_phi), torch.cos(d_phi))

    # d_phi_mod = torch.remainder(d_phi + math.pi, 2 * math.pi) - math.pi
    # d_phi_mod_abs = torch.abs(d_phi_mod)
    # phi_dist = torch.minimum(d_phi_mod_abs, 2 * math.pi - d_phi_mod_abs)

    real1, imag1 = fft_gt.real, fft_gt.imag
    real2, imag2 = fft_rendered.real, fft_rendered.imag
    phi_dist = torch.atan2(real1 * imag2 - real2 * imag1, real1 * real2 + imag1 * imag2)

    return phi_dist
