import argparse
import math
import os
from math import pi

import matplotlib.pyplot as plt
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torchvision.utils import save_image

import config as config
from UD_SfPNet import NetWork
from utils_window import PATCH, STRIDE, hann2d


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Network Testing')
    parser.add_argument('--model_name', type=str, default=None, help="Path to the pre-trained model file to load, e.g., 'xxx.pth'")
    parser.add_argument('--test_batch_size', type=int, default=4, help='Global test batch size (per-process batch size * nprocs)')
    parser.add_argument('--ckpt_path', type=str, default='./pt/UD_SfPNet/1000.pth', help="Path to model weights, e.g., './pt/1000.pth'")
    parser.add_argument('--results_dir', type=str, default='./results_sfp', help='Directory to save predicted normal maps')
    parser.add_argument('--error_maps_dir', type=str, default='./error_maps', help='Directory to save angular error heatmaps')
    parser.add_argument('--summary_path', type=str, default='./table1_metrics.txt', help='Path to save final Table-1-style metrics')
    parser.add_argument('--nprocs', type=int, default=1, help='Number of GPUs to use (1 is recommended for strict benchmark reporting)')
    parser.add_argument('--median_bin_deg', type=float, default=1e-3, help='Histogram bin size in degree for global median estimation')

    args = parser.parse_args()
    if args.median_bin_deg <= 0:
        raise ValueError('--median_bin_deg must be > 0')
    if args.test_batch_size <= 0:
        raise ValueError('--test_batch_size must be >= 1')
    if args.nprocs <= 0:
        raise ValueError('--nprocs must be >= 1')
    return args


def hist_median(hist, total_count, bin_size_deg):
    """Compute median from global histogram counts."""
    cdf = torch.cumsum(hist, dim=0)

    def value_at_rank(rank_zero_based):
        target = torch.tensor(rank_zero_based + 1, dtype=torch.long)
        idx = int(torch.searchsorted(cdf, target).item())
        return idx * bin_size_deg

    if total_count % 2 == 1:
        return value_at_rank(total_count // 2)
    lower = value_at_rank(total_count // 2 - 1)
    upper = value_at_rank(total_count // 2)
    return 0.5 * (lower + upper)


def build_summary_lines(metrics):
    return [
        'Table-1 style quantitative metrics (global pixel aggregation within mask):',
        f"Images evaluated: {metrics['images']}",
        f"Valid pixels: {metrics['valid_pixels']}",
        f"Mean angular error (deg): {metrics['mean']:.4f}",
        f"Median angular error (deg): {metrics['median']:.4f}",
        f"RMSE (deg): {metrics['rmse']:.4f}",
        f"Accuracy < 11.25 deg (%): {metrics['acc11']:.4f}",
        f"Accuracy < 22.5 deg (%): {metrics['acc22']:.4f}",
        f"Accuracy < 30.0 deg (%): {metrics['acc30']:.4f}",
    ]


def save_summary(summary_path, lines):
    summary_dir = os.path.dirname(summary_path)
    if summary_dir:
        os.makedirs(summary_dir, exist_ok=True)
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


def main():
    args = parse_args()

    available_gpus = torch.cuda.device_count()
    if available_gpus == 0:
        raise RuntimeError('No CUDA device found. This script currently requires at least one GPU.')
    if args.nprocs > available_gpus:
        raise ValueError(f'--nprocs={args.nprocs} exceeds available GPUs ({available_gpus})')
    if args.test_batch_size % args.nprocs != 0:
        raise ValueError('--test_batch_size must be divisible by --nprocs')

    # Spawn distributed processes (nprocs=1 is recommended for strict benchmark reporting).
    mp.spawn(main_worker, nprocs=args.nprocs, args=(args.nprocs, args))


def main_worker(local_rank, nprocs, args):
    args.local_rank = local_rank
    config.init_distributed(local_rank=args.local_rank, nprocs=args.nprocs)

    model = NetWork().cuda(args.local_rank)
    checkpoint = torch.load(args.ckpt_path, map_location=f'cuda:{local_rank}')
    model.load_state_dict(checkpoint['model'])

    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.error_maps_dir, exist_ok=True)

    model = config.wrap_model_distributed(model, local_rank=local_rank)
    model.eval()

    test_loader, _ = config.test_dataloaders(args)

    device = torch.device(f'cuda:{local_rank}')
    window = hann2d(PATCH, device).unsqueeze(0).unsqueeze(0)

    # stats: [valid_count, sum_err, sum_err_sq, acc11_count, acc22_count, acc30_count, image_count]
    stats = torch.zeros(7, dtype=torch.float64, device=device)

    num_hist_bins = int(round(180.0 / args.median_bin_deg)) + 1
    median_hist = torch.zeros(num_hist_bins, dtype=torch.long, device=device)

    with torch.no_grad():
        for sample in test_loader:
            inputs = sample['input'].cuda(device)
            image   = sample['image'].cuda(device)
            gt = sample['ground_truth'].float().cuda(device) / 255.0
            mask = sample['mask'].unsqueeze(1).cuda(device)
            gt *= mask
            filenames = sample['filename']

            batch_size = inputs.shape[0]
            _, _, H, W = inputs.shape

            # ----------- Prepare empty containers -----------
            out_sum = torch.zeros(batch_size, 3, H, W, device=device)
            w_sum = torch.zeros(batch_size, 1, H, W, device=device)

            # ----------- Sliding window inference -----------
            for y in range(0, H - PATCH + 1, STRIDE):
                for x in range(0, W - PATCH + 1, STRIDE):
                    patch = inputs[..., y:y + PATCH, x:x + PATCH]
                    patch2 = image[..., y:y+PATCH, x:x+PATCH]
                    pred = model(patch)
                    pred = pred * window
                    out_sum[..., y:y + PATCH, x:x + PATCH] += pred
                    w_sum[..., y:y + PATCH, x:x + PATCH] += window

            # ----------- Reconstruct full prediction -----------
            full_pred = out_sum / w_sum.clamp_min(1e-6)
            full_pred = torch.nn.functional.normalize(full_pred, dim=1)
            full_pred *= mask

            # ----------- Angular error map (degree) -----------
            gt_n = (gt * 2.0 - 1.0) * mask
            # Keep the same angular definition as the original script:
            # cosine similarity is computed per-pixel along channel dimension.
            cos = torch.nn.functional.cosine_similarity(full_pred, gt_n, dim=1, eps=1e-8).unsqueeze(1)
            cos = torch.clamp(cos, -1.0, 1.0)
            ang = torch.acos(cos) * 180.0 / pi
            ang = ang * mask

            # ----------- Global Table-1 style metrics -----------
            valid = ang[mask.bool()]
            if valid.numel() > 0:
                valid64 = valid.double()
                stats[0] += float(valid64.numel())
                stats[1] += valid64.sum()
                stats[2] += (valid64 * valid64).sum()
                stats[3] += (valid < 11.25).sum().double()
                stats[4] += (valid < 22.5).sum().double()
                stats[5] += (valid < 30.0).sum().double()

                quantized = torch.round(valid / args.median_bin_deg).long().clamp(0, num_hist_bins - 1)
                median_hist += torch.bincount(quantized, minlength=num_hist_bins)

            stats[6] += float(batch_size)

            # ----------- Save normal maps + per-image heatmaps -----------
            for b in range(batch_size):
                b_mask = mask[b:b + 1]
                b_ang = ang[b:b + 1]
                b_valid = b_ang[b_mask.bool()]
                if b_valid.numel() == 0:
                    continue

                b_mae = b_valid.mean()
                b_median = b_valid.median()
                b_rmse = torch.sqrt((b_valid ** 2).mean())

                b_filename = filenames[b]
                pred_img = ((full_pred[b:b + 1] + 1.0) * 0.5) * b_mask
                save_image(pred_img, f'{args.results_dir}/{b_filename}_{b_mae.item():.4f}.bmp')

                text = (
                    f'Mean (MAE): {b_mae.item():.2f} deg\n'
                    f'Median   : {b_median.item():.2f} deg\n'
                    f'RMSE     : {b_rmse.item():.2f} deg'
                )

                theta_max = 50.0
                ang_clamped = torch.clamp(b_ang.squeeze(0).squeeze(0), 0.0, theta_max)
                ang_clamped[b_mask.squeeze(0).squeeze(0) == 0] = float('nan')

                fig, ax = plt.subplots(figsize=(6, 6))
                im = ax.imshow(ang_clamped.cpu().numpy(), cmap='jet', vmin=0, vmax=theta_max)
                ax.axis('off')
                ax.text(
                    0.02, 0.98, text,
                    transform=ax.transAxes,
                    ha='left', va='top',
                    fontsize=12, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='black', alpha=0.5, pad=0.4),
                    color='white',
                    zorder=10,
                )
                ax.set_title('Angular Error')

                cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label('Error (deg)')

                fig.tight_layout()
                fig.savefig(f'{args.error_maps_dir}/{b_filename}.png', dpi=300, bbox_inches='tight')
                plt.close(fig)

    # ----------- DDP reduction -----------
    if dist.is_initialized():
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        dist.all_reduce(median_hist, op=dist.ReduceOp.SUM)

    if local_rank == 0:
        total_valid = int(round(stats[0].item()))
        total_images = int(round(stats[6].item()))

        if total_valid == 0:
            lines = [
                'Table-1 style quantitative metrics:',
                'No valid pixels were found inside mask. Please check data/mask.'
            ]
        else:
            mean = stats[1].item() / total_valid
            rmse = math.sqrt(stats[2].item() / total_valid)
            median = hist_median(median_hist.cpu(), total_valid, args.median_bin_deg)
            acc11 = stats[3].item() * 100.0 / total_valid
            acc22 = stats[4].item() * 100.0 / total_valid
            acc30 = stats[5].item() * 100.0 / total_valid

            metrics = {
                'images': total_images,
                'valid_pixels': total_valid,
                'mean': mean,
                'median': median,
                'rmse': rmse,
                'acc11': acc11,
                'acc22': acc22,
                'acc30': acc30,
            }
            lines = build_summary_lines(metrics)

        print('\n' + '\n'.join(lines))
        save_summary(args.summary_path, lines)

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
