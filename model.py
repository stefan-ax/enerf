import os
from typing import Optional, Tuple, List, Union, Callable
from datetime import datetime

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import trange

from utils import *


class PositionalEncoder(nn.Module):
    r"""
    Sine-cosine positional encoder for input points.
    """

    def __init__(
            self,
            d_input: int,
            n_freqs: int,
            log_space: bool = False
    ):
        super().__init__()
        self.d_input = d_input
        self.n_freqs = n_freqs
        self.log_space = log_space
        self.d_output = d_input * (1 + 2 * self.n_freqs)
        self.embed_fns = [lambda x: x]

        # Define frequencies in either linear or log scale
        if self.log_space:
            freq_bands = 2. ** torch.linspace(0., self.n_freqs - 1, self.n_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** (self.n_freqs - 1), self.n_freqs)

        # Alternate sin and cos
        for freq in freq_bands:
            self.embed_fns.append(lambda x, freq=freq: torch.sin(x * freq))
            self.embed_fns.append(lambda x, freq=freq: torch.cos(x * freq))

    def forward(
            self,
            x
    ) -> torch.Tensor:
        r"""
        Apply positional encoding to input.
        """
        return torch.concat([fn(x) for fn in self.embed_fns], dim=-1)


class NeRF(nn.Module):
    r"""
    Neural radiance fields module.
    """

    def __init__(
            self,
            d_input: int = 3,
            n_layers: int = 8,
            d_filter: int = 256,
            skip: Tuple[int] = (4,),
            d_viewdirs: Optional[int] = None,
            d_output: int = 1
    ):
        super().__init__()
        self.d_input = d_input
        self.skip = skip
        self.act = nn.functional.relu
        self.d_viewdirs = d_viewdirs

        # Create model layers
        self.layers = nn.ModuleList(
            [nn.Linear(self.d_input, d_filter)] +
            [nn.Linear(d_filter + self.d_input, d_filter) if i in skip \
                 else nn.Linear(d_filter, d_filter) for i in range(n_layers - 1)]
        )

        # Bottleneck layers
        if self.d_viewdirs is not None:
            # If using viewdirs, split alpha and RGB
            self.alpha_out = nn.Linear(d_filter, 1)
            self.rgb_filters = nn.Linear(d_filter, d_filter)
            self.branch = nn.Linear(d_filter + self.d_viewdirs, d_filter // 2)
            self.output = nn.Linear(d_filter // 2, d_output)
        else:
            # If no viewdirs, use simpler output
            self.output = nn.Linear(d_filter, d_output + 1)

    def forward(
            self,
            x: torch.Tensor,
            viewdirs: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        r"""
        Forward pass with optional view direction.
        """

        # Cannot use viewdirs if instantiated with d_viewdirs = None
        if self.d_viewdirs is None and viewdirs is not None:
            raise ValueError('Cannot input x_direction if d_viewdirs was not given.')

        # Apply forward pass up to bottleneck
        x_input = x
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x))
            if i in self.skip:
                x = torch.cat([x, x_input], dim=-1)

        # Apply bottleneck
        if self.d_viewdirs is not None:
            # Split alpha from network output
            alpha = self.alpha_out(x)

            # Pass through bottleneck to get RGB
            x = self.rgb_filters(x)
            x = torch.concat([x, viewdirs], dim=-1)
            x = self.act(self.branch(x))
            x = self.output(x)

            # Concatenate alphas to output
            x = torch.concat([x, alpha], dim=-1)
        else:
            # Simple output
            x = self.output(x)
        return x


def cumprod_exclusive(
        tensor: torch.Tensor
) -> torch.Tensor:
    r"""
    (Courtesy of https://github.com/krrish94/nerf-pytorch)

    Mimick functionality of tf.math.cumprod(..., exclusive=True), as it isn't available in PyTorch.

    Args:
    tensor (torch.Tensor): Tensor whose cumprod (cumulative product, see `torch.cumprod`) along dim=-1
      is to be computed.
    Returns:
    cumprod (torch.Tensor): cumprod of Tensor along dim=-1, mimiciking the functionality of
      tf.math.cumprod(..., exclusive=True) (see `tf.math.cumprod` for details).
    """

    # Compute regular cumprod first (this is equivalent to `tf.math.cumprod(..., exclusive=False)`).
    cumprod = torch.cumprod(tensor, -1)
    # "Roll" the elements along dimension 'dim' by 1 element.
    cumprod = torch.roll(cumprod, 1, -1)
    # Replace the first element by "1" as this is what tf.cumprod(..., exclusive=True) does.
    cumprod[..., 0] = 1.

    return cumprod


def raw2outputs(
        raw: torch.Tensor,
        z_vals: torch.Tensor,
        rays_d: torch.Tensor,
        raw_noise_std: float = 0.0,
        white_bkgd: bool = False,
        d_output: int = 3,
        log_intensity: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""
    Convert the raw NeRF output into RGB and other maps.
    """

    # Difference between consecutive elements of `z_vals`. [n_rays, n_samples]
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat([dists, 1e10 * torch.ones_like(dists[..., :1])], dim=-1)

    # Multiply each distance by the norm of its corresponding direction ray
    # to convert to real world distance (accounts for non-unit directions).
    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

    # Add noise to model's predictions for density. Can be used to
    # regularize network during training (prevents floater artifacts).
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[..., d_output].shape) * raw_noise_std

    # Predict density of each sample along each ray. Higher values imply
    # higher likelihood of being absorbed at this point. [n_rays, n_samples]
    alpha = 1.0 - torch.exp(-nn.functional.relu(raw[..., d_output] + noise) * dists)

    # Compute weight for RGB of each sample along each ray. [n_rays, n_samples]
    # The higher the alpha, the lower subsequent weights are driven.
    weights = alpha * cumprod_exclusive(1. - alpha + 1e-10)

    # Compute weighted RGB map.
    if log_intensity:
        rgb = torch.relu(raw[..., :d_output])
    else:
        rgb = torch.sigmoid(raw[..., :d_output])  # [n_rays, n_samples, 3]
    rgb_map = torch.sum(weights[..., None] * rgb, dim=-2)  # [n_rays, 3]

    # Estimated depth map is predicted distance.
    depth_map = torch.sum(weights * z_vals, dim=-1)

    # Disparity map is inverse depth.
    disp_map = 1. / torch.max(1e-10 * torch.ones_like(depth_map),
                              depth_map / torch.sum(weights, -1))

    # Sum of weights along each ray. In [0, 1] up to numerical error.
    acc_map = torch.sum(weights, dim=-1)

    # To composite onto a white background, use the accumulated alpha map.
    if white_bkgd:
        rgb_map = rgb_map + (1. - acc_map[..., None])

    return rgb_map, depth_map, acc_map, weights


def nerf_forward(
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        near: float,
        far: float,
        encoding_fn: Callable[[torch.Tensor], torch.Tensor],
        coarse_model: nn.Module,
        kwargs_sample_stratified: dict = None,
        n_samples_hierarchical: int = 0,
        kwargs_sample_hierarchical: dict = None,
        fine_model=None,
        viewdirs_encoding_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        chunksize: int = 2 ** 15,
        d_output=3,
        log_intensity = False
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    r"""
    Compute forward pass through model(s).
    """

    # Set no kwargs if none are given.
    if kwargs_sample_stratified is None:
        kwargs_sample_stratified = {}
    if kwargs_sample_hierarchical is None:
        kwargs_sample_hierarchical = {}

    # Sample query points along each ray.
    query_points, z_vals = sample_stratified(
        rays_o, rays_d, near, far, **kwargs_sample_stratified)

    # Prepare batches.
    batches = prepare_chunks(query_points, encoding_fn, chunksize=chunksize)
    if viewdirs_encoding_fn is not None:
        batches_viewdirs = prepare_viewdirs_chunks(query_points, rays_d,
                                                   viewdirs_encoding_fn,
                                                   chunksize=chunksize)
    else:
        batches_viewdirs = [None] * len(batches)

    # Coarse model pass.
    # Split the encoded points into "chunks", run the model on all chunks, and
    # concatenate the results (to avoid out-of-memory issues).
    predictions = []
    for batch, batch_viewdirs in zip(batches, batches_viewdirs):
        predictions.append(coarse_model(batch, viewdirs=batch_viewdirs))
    raw = torch.cat(predictions, dim=0)
    raw = raw.reshape(list(query_points.shape[:2]) + [raw.shape[-1]])

    # Perform differentiable volume rendering to re-synthesize the RGB image.
    rgb_map, depth_map, acc_map, weights = raw2outputs(raw, z_vals, rays_d,
                                                       d_output=d_output, log_intensity=log_intensity)
    # rgb_map, depth_map, acc_map, weights = render_volume_density(raw, rays_o, z_vals)
    outputs = {
        'z_vals_stratified': z_vals
    }

    # Fine model pass.
    if n_samples_hierarchical > 0:
        # Save previous outputs to return.
        rgb_map_0, depth_map_0, acc_map_0 = rgb_map, depth_map, acc_map

        # Apply hierarchical sampling for fine query points.
        query_points, z_vals_combined, z_hierarch = sample_hierarchical(
            rays_o, rays_d, z_vals, weights, n_samples_hierarchical,
            **kwargs_sample_hierarchical)

        # Prepare inputs as before.
        batches = prepare_chunks(query_points, encoding_fn, chunksize=chunksize)
        if viewdirs_encoding_fn is not None:
            batches_viewdirs = prepare_viewdirs_chunks(query_points, rays_d,
                                                       viewdirs_encoding_fn,
                                                       chunksize=chunksize)
        else:
            batches_viewdirs = [None] * len(batches)

        # Forward pass new samples through fine model.
        fine_model = fine_model if fine_model is not None else coarse_model
        predictions = []
        for batch, batch_viewdirs in zip(batches, batches_viewdirs):
            predictions.append(fine_model(batch, viewdirs=batch_viewdirs))
        raw = torch.cat(predictions, dim=0)
        raw = raw.reshape(list(query_points.shape[:2]) + [raw.shape[-1]])

        # Perform differentiable volume rendering to re-synthesize the RGB image.
        rgb_map, depth_map, acc_map, weights = raw2outputs(raw, z_vals_combined, rays_d,
                                                           d_output=d_output, log_intensity=log_intensity)

        # Store outputs.
        outputs['z_vals_hierarchical'] = z_hierarch
        outputs['rgb_map_0'] = rgb_map_0
        outputs['depth_map_0'] = depth_map_0
        outputs['acc_map_0'] = acc_map_0

    # Store outputs.
    outputs['rgb_map'] = rgb_map
    outputs['depth_map'] = depth_map
    outputs['acc_map'] = acc_map
    outputs['weights'] = weights
    return outputs


def init_models(config):
    r"""
  Initialize models, encoders, and optimizer for NeRF training.
  """
    # Encoders
    encoder = PositionalEncoder(config['d_input'], config['n_freqs'], log_space=config['log_space'])
    encode = lambda x: encoder(x)

    # View direction encoders
    if config['use_viewdirs']:
        encoder_viewdirs = PositionalEncoder(config['d_input'], config['n_freqs_views'],
                                             log_space=config['log_space'])
        encode_viewdirs = lambda x: encoder_viewdirs(x)
        d_viewdirs = encoder_viewdirs.d_output
    else:
        encode_viewdirs = None
        d_viewdirs = None

    # Models
    model = NeRF(encoder.d_output, n_layers=config['n_layers'], d_filter=config['d_filter'], skip=config['skip'],
                 d_viewdirs=d_viewdirs, d_output=config['d_output'])
    model.to(config['device'])
    model_params = list(model.parameters())
    if config['use_fine_model']:
        fine_model = NeRF(encoder.d_output, n_layers=config['n_layers'], d_filter=config['d_filter'],
                          skip=config['skip'],
                          d_viewdirs=d_viewdirs)
        fine_model.to(config['device'])
        model_params = model_params + list(fine_model.parameters())
    else:
        fine_model = None

    # Optimizer
    optimizer = torch.optim.Adam(model_params, lr=config['lr'])

    # Early Stopping
    warmup_stopper = EarlyStopping(patience=50)

    return model, fine_model, encode, encode_viewdirs, optimizer, warmup_stopper


def train(model, fine_model, optimizer, warmup_stopper,
          images, poses, focal,
          testimg, testpose,
          encode, encode_viewdirs,
          config):
    r"""
    Launch training session for NeRF.
    """
    # images.shape = [N, 240, 180]
    # poses.shape = [N, 2, 4, 4]
    # testimg.shape = [240, 180]
    # testpose.shape = [2, 4, 4]

    train_psnrs = []
    train_mses = []
    val_psnrs = []
    val_mses = []
    iternums = []

    if config['save']:
        now = datetime.now()
        dt = now.strftime("%d_%m_%Y_%H_%M")
        save_folder = os.path.join(config['log_path'], os.path.basename(config['data_path'][:-4]) + '_' + dt)
        os.makedirs(save_folder, exist_ok=True)

    for i in trange(config['n_iters']):
        model.train()

        target_img_idx = np.random.randint(images.shape[0])
        target_img = images[target_img_idx].to(config['device'])  # Shape [240, 180]
        if config['center_crop'] and i < config['center_crop_iters']:
            target_img = crop_center(target_img)

        height, width = target_img.shape[:2]
        target_pose_pair = poses[target_img_idx].to(config['device'])  # Shape [2, 4, 4]

        rays_o_t0, rays_d_t0 = get_rays(height, width, focal, target_pose_pair[0])
        rays_o_t1, rays_d_t1 = get_rays(height, width, focal, target_pose_pair[1])

        rays_o_t0 = rays_o_t0.reshape([-1, 3])
        rays_d_t0 = rays_d_t0.reshape([-1, 3])
        rays_o_t1 = rays_o_t1.reshape([-1, 3])
        rays_d_t1 = rays_d_t1.reshape([-1, 3])

        # Run NeRF model
        outputs_t0 = nerf_forward(rays_o_t0, rays_d_t0,
                                  config['near'], config['far'], encode, model,
                                  kwargs_sample_stratified=config['kwargs_sample_stratified'],
                                  n_samples_hierarchical=config['n_samples_hierarchical'],
                                  kwargs_sample_hierarchical=config['kwargs_sample_hierarchical'],
                                  fine_model=fine_model,
                                  viewdirs_encoding_fn=encode_viewdirs,
                                  chunksize=config['chunksize'],
                                  d_output=config['d_output'],
                                  log_intensity=config['log_intensity'])

        outputs_t1 = nerf_forward(rays_o_t1, rays_d_t1,
                                  config['near'], config['far'], encode, model,
                                  kwargs_sample_stratified=config['kwargs_sample_stratified'],
                                  n_samples_hierarchical=config['n_samples_hierarchical'],
                                  kwargs_sample_hierarchical=config['kwargs_sample_hierarchical'],
                                  fine_model=fine_model,
                                  viewdirs_encoding_fn=encode_viewdirs,
                                  chunksize=config['chunksize'],
                                  d_output=config['d_output'],
                                  log_intensity=config['log_intensity'])

        # Check for any numerical issues.
        for k, v in outputs_t0.items():
            if torch.isnan(v).any():
                print(f"! [Numerical Alert] {k} contains NaN.")
            if torch.isinf(v).any():
                print(f"! [Numerical Alert] {k} contains Inf.")

        for k, v in outputs_t1.items():
            if torch.isnan(v).any():
                print(f"! [Numerical Alert] {k} contains NaN.")
            if torch.isinf(v).any():
                print(f"! [Numerical Alert] {k} contains Inf.")

        # Backprop!
        predicted_image_t0 = outputs_t0['rgb_map']
        predicted_image_t1 = outputs_t1['rgb_map']

        if config['log_intensity']:
            predicted_image_t0[predicted_image_t0 != 0] = torch.log(predicted_image_t0[predicted_image_t0 != 0])
            predicted_image_t1[predicted_image_t1 != 0] = torch.log(predicted_image_t1[predicted_image_t1 != 0])

        # plt.imshow(torch.clone(rgb_predicted).cpu().detach().numpy().reshape(50, 50, 3))
        # plt.imshow(torch.clone(predicted_image_t0).cpu().detach().numpy().reshape(height, width, 1))
        loss = torch.nn.functional.mse_loss(predicted_image_t1[:, 0] - predicted_image_t0[:, 0],
                                            target_img.reshape(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Compute mean-squared error between predicted and target images.
        psnr = -10. * torch.log10(loss)
        train_psnrs.append(psnr.item())
        train_mses.append(loss.item())
        # print(psnr.item())

        # Evaluate testimg at given display rate.
        if i % config['display_rate'] == 0 and config['display'] and i > 0:
            model.eval()
            height, width = testimg.shape[:2]
            rays_o_t0, rays_d_t0 = get_rays(height, width, focal, testpose[0])
            rays_o_t1, rays_d_t1 = get_rays(height, width, focal, testpose[1])

            rays_o_t0 = rays_o_t0.reshape([-1, 3])
            rays_d_t0 = rays_d_t0.reshape([-1, 3])
            rays_o_t1 = rays_o_t1.reshape([-1, 3])
            rays_d_t1 = rays_d_t1.reshape([-1, 3])

            test_outputs = []
            for rays_o, rays_d in zip([rays_o_t0, rays_d_t0], [rays_o_t1, rays_d_t1]):
                with torch.no_grad():
                    test_outputs.append(nerf_forward(rays_o, rays_d,
                                                     config['near'], config['far'], encode, model,
                                                     kwargs_sample_stratified=config['kwargs_sample_stratified'],
                                                     n_samples_hierarchical=config['n_samples_hierarchical'],
                                                     kwargs_sample_hierarchical=config['kwargs_sample_hierarchical'],
                                                     fine_model=fine_model,
                                                     viewdirs_encoding_fn=encode_viewdirs,
                                                     chunksize=config['chunksize'],
                                                     d_output=config['d_output'],
                                                     log_intensity=config['log_intensity']))

            rgbs_predicted = [out['rgb_map'] for out in test_outputs]
            losses = [torch.nn.functional.mse_loss(rgb_predicted[:, 0], testimg.reshape(-1))
                      for rgb_predicted in rgbs_predicted]
            # print("Losses: --- ", losses[0].item(), ' --- ', losses[1].item())
            val_psnr = sum([-10. * torch.log10(loss) for loss in losses])/2
            val_mse = sum(losses)/2

            val_psnrs.append(val_psnr.item())
            val_mses.append(val_mse.item())
            iternums.append(i)

            # Plot example outputs
            fig, ax = plt.subplots(1, 4, figsize=(24, 4), gridspec_kw={'width_ratios': [1, 1, 1, 3]})
            if config['log_intensity']:
                ax[0].imshow(np.exp(rgbs_predicted[0].reshape([height, width]).detach().cpu().numpy()))
            else:
                ax[0].imshow(rgbs_predicted[0].reshape([height, width]).detach().cpu().numpy())
            ax[0].set_title("Test pose 0")
            ax[0].set_xlabel(f'Iteration: {i}')
            ax[0].set_axis_off()

            if config['log_intensity']:
                ax[1].imshow(np.exp(rgbs_predicted[1].reshape([height, width]).detach().cpu().numpy()))
            else:
                ax[1].imshow(rgbs_predicted[1].reshape([height, width]).detach().cpu().numpy())
            ax[1].set_title(f'Test pose 1')
            ax[1].set_axis_off()

            ax[2].plot(range(0, i + 1), train_mses, 'r', label='train')
            ax[2].plot(iternums, val_mses, 'b', label='blue')
            ax[2].set_title('MSE')
            ax[2].set_yscale('log')
            ax[2].legend()

            z_vals_strat = test_outputs[0]['z_vals_stratified'].view((-1, config['n_samples']))
            z_sample_strat = z_vals_strat[z_vals_strat.shape[0] // 2].detach().cpu().numpy()
            if 'z_vals_hierarchical' in test_outputs[0]:
                z_vals_hierarch = test_outputs[0]['z_vals_hierarchical'].view((-1, config['n_samples_hierarchical']))
                z_sample_hierarch = z_vals_hierarch[z_vals_hierarch.shape[0] // 2].detach().cpu().numpy()
            else:
                z_sample_hierarch = None
            _ = plot_samples(z_sample_strat, z_sample_hierarch, ax=ax[3])
            ax[3].margins(0)

            if config['save']:
                plt.savefig(os.path.join(save_folder, f'plot_{i}_it.png'), bbox_inches='tight')
            else:
                plt.show()

        # Check PSNR for issues and stop if any are found.
        if i == config['warmup_iters'] - 1:
            if val_psnr < config['warmup_min_fitness']:
                print(f"Val PSNR {val_psnr} below warmup_min_fitness {config['warmup_min_fitness']}. Stopping...")
                return False, train_psnrs, val_psnrs
        elif i < config['warmup_iters']:
            if warmup_stopper is not None and warmup_stopper(i, psnr):
                print(f'Train PSNR flatlined at {psnr} for {warmup_stopper.patience} iters. Stopping...')
                return False, train_psnrs, val_psnrs

    return True, train_psnrs, val_psnrs
