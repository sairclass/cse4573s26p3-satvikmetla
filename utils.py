'''
Helper and zip functions.
Please read the instructions before you start task2.

Please do NOT make any change to this file.
'''

import os, torch
import zipfile, argparse
import matplotlib.pyplot as plt
from torchvision.transforms import functional as F
import torchvision.io as io


def show_image(img: torch.Tensor, delay=1000):
    """Shows an image.
    """
    plt.imshow(F.to_pil_image(img))
    plt.show()

def read_image(img_path):
    return io.read_image(img_path)

def read_images(img_dir):
    res = {}
    for img_name in sorted(os.listdir(img_dir)):
        img_path = os.path.join(img_dir, img_name)
        img = read_image(img_path)
        res[img_name] = img
    return res

def write_image(input_image: torch.Tensor, output_path: str):
    io.write_png(input_image, output_path)


def bgr_to_rgb(img_bgr):
    """
    Converts an image from BGR format to RGB format.

    Args:
        img_bgr (torch.Tensor): A BGR image tensor with shape (C, H, W) or (B, C, H, W),
                                 where C = 3 (BGR channels), H = height, W = width, and B = batch size.

    Returns:
        torch.Tensor: An RGB image tensor with shape (C, H, W) or (B, C, H, W).
    """
    assert img_bgr.dim() == 3 or img_bgr.dim() == 4, "Input tensor must have either 3 or 4 dimensions (C, H, W) or (B, C, H, W)"
    assert img_bgr.size(-3) == 3, "Input tensor must have 3 channels (BGR)"

    # Rearrange the BGR channels to RGB by flipping the channel dimension.
    img_rgb = torch.flip(img_bgr, dims=(-3,))

    return img_rgb

def parse_args():
    parser = argparse.ArgumentParser(description="CSE 473/573 project 2 submission.")
    parser.add_argument("--ubit", type=str)
    args = parser.parse_args()
    return args

def files2zip(files: list, zip_file_name: str):
    with zipfile.ZipFile(zip_file_name, mode='w', compression=zipfile.ZIP_DEFLATED) as zf:
        for file in files:
            path, name = os.path.split(file)
            if os.path.exists(file):
                zf.write(file, arcname=name)
            else:
                print('Zipping error! Your submission must have file %s, even if you does not change that.' % name)


if __name__ == "__main__":
    args = parse_args()
    file_list = ['face.py', 'result_task1.json','result_task1_val.json', 'result_task2.json', 'viz_task1.png', 'viz_task2.png']
    files2zip(file_list, 'submission_' + args.ubit + '.zip')