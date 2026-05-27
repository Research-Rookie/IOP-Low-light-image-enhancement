from tqdm import tqdm
import torch
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import cv2
import lpips
from skimage.metrics import peak_signal_noise_ratio as psnr_loss
from skimage.metrics import structural_similarity as ssim_loss
import argparse
import csv

parser = argparse.ArgumentParser(description='PSNR SSIM script', add_help=False)
parser.add_argument('--input_images_path', default=r'D:\MN\weiguang\DMPHN-cvpr19-master-master\scie\EN')
parser.add_argument('--image2smiles2image_save_path', default=r'D:\MN\weiguang\DMPHN-cvpr19-master-master\scie\nd')
parser.add_argument('-v', '--version', type=str, default='0.1')
args = parser.parse_args()


def is_png_file(filename):
    return any(filename.endswith(extension) for extension in [".jpg", ".png", ".jpeg"])


def load_img(filepath):
    img = cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    img = img / 255.
    return img


class DataLoaderVal(Dataset):
    def __init__(self, target_transform=None):
        super(DataLoaderVal, self).__init__()

        self.target_transform = target_transform

        gt_dir = args.input_images_path
        input_dir = args.image2smiles2image_save_path

        clean_files = sorted(os.listdir(os.path.join(gt_dir)))
        noisy_files = sorted(os.listdir(os.path.join(input_dir)))

        self.clean_filenames = [os.path.join(gt_dir, x) for x in clean_files if is_png_file(x)]
        self.noisy_filenames = [os.path.join(input_dir, x) for x in noisy_files if is_png_file(x)]

        self.tar_size = len(self.clean_filenames)

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        tar_index = index % self.tar_size

        clean = torch.from_numpy(np.float32(load_img(self.clean_filenames[tar_index])))
        noisy = torch.from_numpy(np.float32(load_img(self.noisy_filenames[tar_index])))

        clean_filename = os.path.split(self.clean_filenames[tar_index])[-1]
        noisy_filename = os.path.split(self.noisy_filenames[tar_index])[-1]

        clean = clean.permute(2, 0, 1)
        noisy = noisy.permute(2, 0, 1)

        return clean, noisy, clean_filename, noisy_filename


def get_validation_data():
    return DataLoaderVal(None)


test_dataset = get_validation_data()
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False)

loss_fn = lpips.LPIPS(net='alex', version=args.version)

if __name__ == '__main__':
    results_table = []
    # ---------------------- PSNR + SSIM ----------------------
    psnr_val_rgb = []
    ssim_val_rgb = []
    for ii, data_test in enumerate(tqdm(test_loader), 0):
        rgb_groundtruth = data_test[0].numpy().squeeze().transpose((1, 2, 0))
        rgb_restored = data_test[1].cuda()

        rgb_restored = torch.clamp(rgb_restored, 0, 1).cpu().numpy().squeeze().transpose((1, 2, 0))
        psnr_score = psnr_loss(rgb_restored, rgb_groundtruth)
        ssim_score = ssim_loss(
            rgb_restored,
            rgb_groundtruth,
            data_range=1.0,
            win_size=127,
            channel_axis=2
        )
        psnr_val_rgb.append(psnr_score)
        ssim_val_rgb.append(ssim_score)

        print(f"[{data_test[2][0]}] PSNR: {psnr_score:.2f}, SSIM: {ssim_score:.4f}")
        results_table.append({
            "filename": data_test[2][0],
            "psnr": psnr_score,
            "ssim": ssim_score,
            "lpips": None
        })
    psnr_val_rgb = sum(psnr_val_rgb) / len(test_dataset)
    ssim_val_rgb = sum(ssim_val_rgb) / len(test_dataset)

    # ---------------------- LPIPS ----------------------
    files = os.listdir(args.input_images_path)
    i = 0
    total_lpips_distance = 0
    average_lpips_distance = 0
    for file in files:

        try:
            # Load images
            img0 = lpips.im2tensor(lpips.load_image(os.path.join(args.input_images_path, file)))
            img1 = lpips.im2tensor(lpips.load_image(os.path.join(args.image2smiles2image_save_path, file)))

            if (os.path.exists(os.path.join(args.input_images_path, file)),
                os.path.exists(os.path.join(args.image2smiles2image_save_path, file))):
                i = i + 1

            # Compute distance
            current_lpips_distance = loss_fn.forward(img0, img1)
            current_score = current_lpips_distance.item()
            total_lpips_distance += current_score
            print(f"[{file}] LPIPS: {current_score:.4f}")
            for row in results_table:
                if row["filename"] == file:
                    row["lpips"] = current_score
                    break

        except Exception as e:
            print(e)

    average_lpips_distance = float(total_lpips_distance) / i

    print("The processed iamges is ", i)
    print("PSNR: %f, SSIM: %f, LPIPS: %f " % (psnr_val_rgb, ssim_val_rgb, average_lpips_distance))
    # 保存 CSV 文件
    csv_save_path = 'OUR.csv'
    with open(csv_save_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Image Name", "PSNR", "SSIM", "LPIPS"])

        for row in results_table:
            writer.writerow([
                row["filename"],
                f"{row['psnr']:.4f}",
                f"{row['ssim']:.4f}",
                f"{row['lpips']:.4f}" if row['lpips'] is not None else "N/A"
            ])

        # 添加平均值行
        writer.writerow([
            "Average",
            f"{psnr_val_rgb:.4f}",
            f"{ssim_val_rgb:.4f}",
            f"{average_lpips_distance:.4f}"
        ])

    print(f"\n✅ All results saved to {csv_save_path}")
