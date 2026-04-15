import os
import sys
import time
import numpy as np
import torch
from torch import nn
import matplotlib
from models import Finer, Siren, Gauss, PEMLP, Wire,Siren_1
matplotlib.use('agg')
import matplotlib.pyplot as plt
from tqdm import trange
from utils import setup_seed, mse_fn, psnr_fn, read_image, write_image
import random
import imageio.v2 as imageio
import configargparse
from pytorch_msssim import ssim
import os
import numpy as np
import imageio
import cv2




setup_seed(3407)
device = torch.device('cuda:0')


class Logger:
    filename = 'experiment_scripts_finer/logs/calc_time/logs_time/logtime.txt'
    
    @staticmethod
    def write(text):
        with open(Logger.filename, 'a') as log_file:
            log_file.write(text + '\n')
    

def get_train_data(img_path, zero_mean=True):
    img = np.array(imageio.imread(img_path), dtype=np.float32) / 255.
    # normalize
    if zero_mean:
        img = (img - 0.5) / 0.5 # [-1, 1]
    H, W, C = img.shape
    gt = torch.tensor(img).view(-1, C)
    coords = torch.stack(torch.meshgrid([torch.linspace(-1, 1, H), torch.linspace(-1, 1, W)], indexing='ij'), dim=-1).view(-1, 2)
    return coords, gt, [H, W, C] 



def get_model(opts):
    if opts.model_type == 'finer':
        model = Finer(in_features=2, out_features=3, hidden_layers=opts.hidden_layers, hidden_features=opts.hidden_features,
                      first_omega_0=opts.first_omega, hidden_omega_0=opts.hidden_omega, first_bias_scale=opts.first_bias_scale, scale_req_grad=opts.scale_req_grad)
    elif opts.model_type == 'siren':
        model = Siren(in_features=2, out_features=3, hidden_layers=opts.hidden_layers, hidden_features=opts.hidden_features,
                      first_omega_0=opts.first_omega, hidden_omega_0=opts.hidden_omega)
    elif opts.model_type == 'wire':
        model = Wire(in_features=2, out_features=3, hidden_layers=opts.hidden_layers, hidden_features=opts.hidden_features,
                     first_omega_0=opts.first_omega, hidden_omega_0=opts.hidden_omega, scale=opts.scale)
    elif opts.model_type == 'gauss':
        model = Gauss(in_features=2, out_features=3, hidden_layers=opts.hidden_layers, hidden_features=opts.hidden_features,
                      scale=opts.scale)
    elif opts.model_type == 'pemlp':
        model = PEMLP(in_features=2, out_features=3, hidden_layers=opts.hidden_layers, hidden_features=opts.hidden_features,
                      N_freqs=opts.N_freqs)
    elif opts.model_type == 'siren_1':
        model = Siren_1(in_features=2, out_features=3, hidden_layers=opts.hidden_layers, hidden_features=opts.hidden_features,
                      first_omega_0=opts.first_omega, hidden_omega_0=opts.hidden_omega, first_bias_scale=opts.first_bias_scale, scale_req_grad=opts.scale_req_grad)
    return model 




def get_opts():
    parser = configargparse.ArgumentParser()
    parser.add_argument('--logdir', type=str, default='logs_siren', help='logdir')
    parser.add_argument('--exp_name', type=str, default='test', help='experiment name')
    
    # dataset
    parser.add_argument('--dataset_dir', type=str, default= '/media/juno/hdd8tb/new/FINER/data/div2k/test_data', help='dataset')
    parser.add_argument('--img_id', type=int, default=0, help='id of image')
    parser.add_argument('--not_zero_mean', action='store_true') 
    
    # training options
    parser.add_argument('--num_epochs', type=int, default=5000, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--steps_til_summary', type=int, default=10, help='steps_til_summary')
    
    # network 
    parser.add_argument('--model_type', type=str, default='siren', required=['siren', 'finer', 'wire', 'gauss', 'pemlp'])
    parser.add_argument('--hidden_layers', type=int, default=3, help='hidden_layers') 
    parser.add_argument('--hidden_features', type=int, default=256, help='hidden_features')
    
    #
    parser.add_argument('--first_omega', type=float, default=30, help='(siren, wire, finer)')    
    parser.add_argument('--hidden_omega', type=float, default=30, help='(siren, wire, finer)')    
    parser.add_argument('--scale', type=float, default=30, help='simga (wire, guass)')    
    parser.add_argument('--N_freqs', type=int, default=10, help='(PEMLP)')    

    # finer
    parser.add_argument('--first_bias_scale', type=float, default=None, help='bias_scale of the first layer')    
    parser.add_argument('--scale_req_grad', action='store_true') 
    return parser.parse_args()


def train(model, coords, gt, size, zero_mean=True, loss_fn=mse_fn, 
          lr=1e-3, num_epochs=2000, steps_til_summary=10):

    optimizer = torch.optim.Adam(lr=lr, params=model.parameters())
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda iter: 0.1 ** min(iter / num_epochs, 1)
    )

    train_iter = []
    train_psnr = []
    grad_norms = []
    loss_history = []
    time_history = []

    total_time = 0

    for epoch in trange(1, num_epochs + 1):

        time_start = time.time()

        pred = model(coords)
        loss = loss_fn(pred, gt)

        optimizer.zero_grad()
        loss.backward()

        # ==========================
        # 🔍 Gradient Norm Monitoring
        # ==========================
        total_grad_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                total_grad_norm += param_norm.item() ** 2
        total_grad_norm = total_grad_norm ** 0.5

        grad_norms.append(total_grad_norm)
        loss_history.append(loss.item())

        # 🚨 Gradient explosion warning
        if total_grad_norm > 300:
            print(f"[WARNING] Large gradient norm at epoch {epoch}: {total_grad_norm:.2f}")

        # 🚨 Loss instability warning
        if epoch > 10:
            recent_loss = np.mean(loss_history[-10:])
            if loss.item() > 5 * recent_loss:
                print(f"[WARNING] Loss instability at epoch {epoch}")

        optimizer.step()
        scheduler.step()

        torch.cuda.synchronize()
        epoch_time = time.time() - time_start
        total_time += epoch_time
        time_history.append(total_time)

        # ==========================
        # 🔍 PSNR Monitoring
        # ==========================
        if not epoch % steps_til_summary:
            with torch.no_grad():

                if zero_mean:
                    current_psnr = psnr_fn(
                        model(coords)/2+0.5,
                        gt/2+0.5
                    ).item()
                else:
                    current_psnr = psnr_fn(
                        model(coords),
                        gt
                    ).item()

                train_iter.append(epoch)
                train_psnr.append(current_psnr)

                # 🚨 PSNR oscillation warning
                if len(train_psnr) > 5:
                    recent_psnr = train_psnr[-5:]
                    if np.std(recent_psnr) > 1.0:
                        print(f"[WARNING] PSNR oscillating at epoch {epoch}")

        # ==========================
        # 🔍 Output Range Check (Every 500 epochs)
        # ==========================
        if epoch % 500 == 0:
            with torch.no_grad():
                pred_img_tmp = model(coords)
                print(
                    f"[Epoch {epoch}] Prediction range: "
                    f"{pred_img_tmp.max().item():.3f} "
                    f"{pred_img_tmp.min().item():.3f}"
                )

    # ==========================
    # Final Reconstruction
    # ==========================
    with torch.no_grad():
        if zero_mean:
            pred_img = model(coords).reshape(size)/2 + 0.5
        else:
            pred_img = model(coords).reshape(size)

    ret_dict = {
        'train_iter': train_iter,
        'train_psnr': train_psnr,
        'grad_norms': grad_norms,
        'loss_history': loss_history,
        'pred_img': pred_img,
        'model_state': model.state_dict(),
        'total_time': total_time,
        'time_history': time_history
    }

    return ret_dict



if __name__ == '__main__':
    opts = get_opts()
    
    print('--- Run Configuration ---')
    for k, v in vars(opts).items():
        print(k, '=', v)
    print('--- Run Configuration ---')
    
    ## logdir 
    logdir = os.path.join(opts.logdir, opts.exp_name)
    os.makedirs(logdir, exist_ok=True)
    
    # image path
    coords, gt, size = get_train_data(os.path.join(opts.dataset_dir, f'%02d.png'%(opts.img_id)), not opts.not_zero_mean)
    coords = coords.to(device)
    gt = gt.to(device)
    print(coords.shape, gt.shape, gt.max(), gt.min())
    
    # model
    model = get_model(opts).to(device)
    print(model)
     
        
    
    # train
    ret_dict = train(model, coords, gt, size, not opts.not_zero_mean, mse_fn, lr=opts.lr, num_epochs=opts.num_epochs, steps_til_summary=opts.steps_til_summary)
    
    # save 
    torch.save(ret_dict, os.path.join(logdir, 'outputs_%02d.pt'%((opts.img_id))))
    
    print('Train PSNR: %.4f'%(ret_dict['train_psnr'][-1]))

    # ==========================
    # Final Metrics Computation
    # ==========================
    with torch.no_grad():

        pred_img = ret_dict['pred_img']

        if not opts.not_zero_mean:
            gt_img = gt.reshape(size)/2 + 0.5
        else:
            gt_img = gt.reshape(size)

        pred_img = torch.clamp(pred_img, 0.0, 1.0)
        gt_img = torch.clamp(gt_img, 0.0, 1.0)

        pred_nchw = pred_img.permute(2, 0, 1).unsqueeze(0)
        gt_nchw = gt_img.permute(2, 0, 1).unsqueeze(0)

        mse = torch.mean((pred_img - gt_img) ** 2)
        final_psnr = -10. * torch.log10(mse)

        final_ssim = ssim(
            pred_nchw,
            gt_nchw,
            data_range=1.0,
            size_average=True
        )

        

    print("\n===== Final Metrics =====")
    print(f"PSNR  : {final_psnr.item():.4f}")
    print(f"SSIM  : {final_ssim.item():.4f}")
    print("=========================\n")

    # 🔥 Save Reconstructed Images
    # ==========================
    save_dir = os.path.join(logdir, "reconstructions_14")
    os.makedirs(save_dir, exist_ok=True)

    # Convert tensor to numpy
    pred_np = pred_img.detach().cpu().numpy()
    gt_np = gt_img.detach().cpu().numpy()

    # Convert to uint8 [0,255]
    pred_uint8 = (pred_np * 255.0).astype(np.uint8)
    gt_uint8 = (gt_np * 255.0).astype(np.uint8)

    # Save images
    imageio.imwrite(os.path.join(save_dir, f"reconstruction_{opts.img_id:02d}.png"), pred_uint8)
    imageio.imwrite(os.path.join(save_dir, f"groundtruth_{opts.img_id:02d}.png"), gt_uint8)

    # Optional: Save absolute error map
    error_map = np.abs(pred_np - gt_np)
    error_map = (error_map / error_map.max() * 255.0).astype(np.uint8)
    imageio.imwrite(os.path.join(save_dir, f"error_map_{opts.img_id:02d}.png"), error_map)

    print(f"Images saved to: {save_dir}")

    # import os
    # import numpy as np
    # import matplotlib.pyplot as plt
    # from scipy.io import savemat

    # # ==========================
    # # 📊 Save Training Graphs
    # # ==========================

    # graph_dir = os.path.join(logdir, "psnr_vs_epoch_7_64")
    # os.makedirs(graph_dir, exist_ok=True)

    # train_iter = ret_dict['train_iter']
    # train_psnr = ret_dict['train_psnr']
    # grad_norms = ret_dict['grad_norms']

    # # --------------------------
    # # PSNR Graph (every 1000 epochs)
    # # --------------------------
    # psnr_epochs = []
    # psnr_values = []

    # for e, p in zip(train_iter, train_psnr):
    #     if e % 1000 == 0:
    #         psnr_epochs.append(e)
    #         psnr_values.append(p)

    # psnr_epochs = np.array(psnr_epochs)
    # psnr_values = np.array(psnr_values)

    # # --------------------------
    # # Save numpy data
    # # --------------------------
    # np_save_path = os.path.join(graph_dir, "psnr_data.npz")

    # np.savez(
    #     np_save_path,
    #     epochs=psnr_epochs,
    #     psnr=psnr_values
    # )

    # print(f"PSNR numpy data saved to: {np_save_path}")

    # # --------------------------
    # # Save MATLAB (.mat) format
    # # --------------------------
    # mat_save_path = os.path.join(graph_dir, "psnr_data.mat")

    # savemat(
    #     mat_save_path,
    #     {
    #         "epochs": psnr_epochs,
    #         "psnr": psnr_values
    #     }
    # )

    # print(f"PSNR mat file saved to: {mat_save_path}")

    # # --------------------------
    # # Plot Graph
    # # --------------------------
    # plt.figure()
    # plt.plot(psnr_epochs, psnr_values, marker='o')
    # plt.xlabel("Epoch")
    # plt.ylabel("PSNR")
    # plt.title("PSNR vs Epoch (every 1000 epochs)")
    # plt.grid(True)

    # psnr_path = os.path.join(graph_dir, "psnr_vs_epoch.png")

    # plt.savefig(psnr_path)
    # plt.close()

    # print(f"PSNR graph saved to: {psnr_path}")
    
    ## Reconstructed images

    # save_dir = os.path.join(logdir, "reconstructions_14")
    # os.makedirs(save_dir, exist_ok=True)

    # # Convert tensor to numpy
    # pred_np = pred_img.detach().cpu().numpy()
    # gt_np = gt_img.detach().cpu().numpy()

    # # Convert to uint8 [0,255]
    # pred_uint8 = (pred_np * 255.0).astype(np.uint8)
    # gt_uint8 = (gt_np * 255.0).astype(np.uint8)

    # # -----------------------------
    # # Fourier Spectrum Function
    # # -----------------------------
    # def compute_fourier(img):

    #     if img.ndim == 3:
    #         img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    #     f = np.fft.fft2(img)
    #     fshift = np.fft.fftshift(f)

    #     magnitude = np.log(np.abs(fshift) + 1)

    #     magnitude = magnitude - magnitude.min()
    #     magnitude = magnitude / magnitude.max()
    #     magnitude = (magnitude * 255).astype(np.uint8)

    #     magnitude = cv2.applyColorMap(magnitude, cv2.COLORMAP_JET)

    #     return magnitude


    # # Compute Fourier spectrum
    # fourier_pred = compute_fourier(pred_uint8)
    # fourier_gt = compute_fourier(gt_uint8)

    # # Resize spectrum for overlay
    # overlay_size = 100
    # fourier_pred = cv2.resize(fourier_pred, (overlay_size, overlay_size))
    # fourier_gt = cv2.resize(fourier_gt, (overlay_size, overlay_size))


    # # -----------------------------
    # # Overlay Fourier on image
    # # -----------------------------
    # def overlay_spectrum(image, spectrum):

    #     img = image.copy()

    #     h, w = img.shape[:2]

    #     img[0:overlay_size, w-overlay_size:w] = spectrum

    #     return img


    # pred_with_spec = overlay_spectrum(pred_uint8, fourier_pred)
    # gt_with_spec = overlay_spectrum(gt_uint8, fourier_gt)


    # # -----------------------------
    # # Save images
    # # -----------------------------
    # imageio.imwrite(
    #     os.path.join(save_dir, f"reconstruction_{opts.img_id:02d}.png"),
    #     pred_with_spec
    # )

    # imageio.imwrite(
    #     os.path.join(save_dir, f"groundtruth_{opts.img_id:02d}.png"),
    #     gt_with_spec
    # )


    # #-----------------------------
    # #Error Map
    # #-----------------------------
    # error_map = np.abs(pred_np - gt_np)
    # error_map = (error_map / error_map.max() * 255.0).astype(np.uint8)

    # imageio.imwrite(
    #     os.path.join(save_dir, f"error_map_{opts.img_id:02d}.png"),
    #     error_map
    # )

    # print(f"Images saved to: {save_dir}")

    
    # [option]
    # Logger.write(f'{opts.exp_name}      %.4f'%(ret_dict['total_time']))