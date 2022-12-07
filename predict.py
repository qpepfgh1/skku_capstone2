import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torchvision import transforms

from utils.data_loading import BasicDataset
from unet import UNet
from unet import UNet_3Plus
from utils.utils import plot_img_and_mask
from applyContour import *

import matplotlib.pyplot as plt

import jsonMethod


def img_Contrast(img):
    # -----Converting image to LAB Color model-----------------------------------
    cv2.imwrite('constrast.jpg', img)
    img = cv2.imread('./constrast.jpg')

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # -----Splitting the LAB image to different channels-------------------------
    l, a, b = cv2.split(lab)

    # -----Applying CLAHE to L-channel-------------------------------------------
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(80, 80))
    cl = clahe.apply(l)

    # -----Merge the CLAHE enhanced L-channel with the a and b channel-----------
    limg = cv2.merge((cl, a, b))

    # -----Converting image from LAB Color model to RGB model--------------------
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    return final

def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)

        if net.n_classes > 1:
            probs = F.sigmoid(output)[0]
        else:
            probs = torch.sigmoid(output)[0]

        tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((full_img.size[1], full_img.size[0])),
            transforms.ToTensor()
        ])

        full_mask = tf(probs.cpu()).squeeze()

    if net.n_classes == 1:
        return (full_mask > out_threshold).numpy()
    else:
        # print(np.shape(full_mask.numpy()))
        # print(np.shape(F.one_hot(full_mask.argmax(dim=0), net.n_classes).numpy()))
        # print(np.shape(F.one_hot(full_mask.argmax(dim=0), net.n_classes).permute(2, 0, 1).numpy()))
        # print(np.shape(full_mask.permute(2,0,1).numpy()))
        # img = full_mask.numpy()
        img = F.one_hot(full_mask.argmax(dim=0), net.n_classes).permute(2, 0, 1).numpy()
        # print(np.unique(img))
        # print(np.shape(img))

        # img0 = F.one_hot(full_mask[0].argmax(dim=0)).numpy()
        # img1 = F.one_hot(full_mask[1].argmax(dim=0)).numpy()
        # img2 = F.one_hot(full_mask[2].argmax(dim=0)).numpy()
        # img3 = F.one_hot(full_mask[3].argmax(dim=0)).numpy()
        return img


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE', help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames of input images')
    parser.add_argument('--output', '-o', metavar='OUTPUT', nargs='+', help='Filenames of output images')
    parser.add_argument('--viz', '-v', action='store_true', help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5, help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=0.3, help='Scale factor for the input images')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--dir', '-d', type=str, dest="dir", default="./data/test/", help='Scale factor for the input images')
    parser.add_argument('--model_name', '-mn', type=str, dest="model_name", default="", help='Scale factor for the input images')

    return parser.parse_args()


def get_output_filenames(args):
    def _generate_name(fn):
        return f'{os.path.splitext(fn)[0]}_OUT.png'

    return args.output or list(map(_generate_name, args.input))


def mask_to_image(mask: np.ndarray):
    if mask.ndim == 2:
        return Image.fromarray((mask * 255).astype(np.uint8))
    elif mask.ndim == 3:
        return Image.fromarray((np.argmax(mask, axis=0) * 255 / mask.shape[0]).astype(np.uint8))


if __name__ == '__main__':
    args = get_args()
    print(args)
    # in_files = args.input
    # out_files = get_output_filenames(args)

    unet_version = 3
    if unet_version == 1:
        net = UNet(n_channels=3, n_classes=4, bilinear=args.bilinear)
    elif unet_version == 3:
        net = UNet_3Plus(n_channels=3, n_classes=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device))

    logging.info('Model loaded!')

    dir = args.dir
    model_name = args.model_name
    MIC1_dir = "./src/result/" + model_name + "/img/MIC1/"
    MIC4_dir = "./src/result/" + model_name + "/img/MIC4/"
    MIC5_dir = "./src/result/" + model_name + "/img/MIC5/"

    print(MIC1_dir, MIC4_dir, MIC5_dir)

    file_list = os.listdir(dir)

    # file_list.sort(key=lambda file: int(file.split("_")[1].split(".")[0]), reverse=True)

    for i in range(len(file_list)):
        jsonData = {}
        jsonData['total'] = len(file_list)
        jsonData['count'] = i+1
        jsonMethod.setData(type="TEST", data=jsonData)

        filename = os.path.join(dir, file_list[i])
        print(f'\nPredicting image {filename} ...')
        img = Image.open(filename)

        mask = predict_img(net=net,
                           full_img=img,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           device=device)

        if not args.no_save:
            mask_back = np.where(mask[0, :, :] == 1, 0, mask[0, :, :])
            mask_true = np.where(mask[1, :, :] == 1, 200, mask[1, :, :])
            mask_false = np.where(mask[2, :, :] == 1, 50, mask[2, :, :])
            mask_tf = np.where(mask[3, :, :] == 1, 255, mask[3, :, :])

            mask = mask_back + mask_true + mask_false + mask_tf
            np.where(mask == 50, 1, mask)
            np.where(mask == 200, 2, mask)
            np.where(mask == 255, 3, mask)

            mask = Image.fromarray((mask * 255).astype(np.uint8))

            img.save(MIC1_dir + file_list[i], file_list[i].split(".")[-1])  # 원본 이미지 저장
            mask.save(MIC4_dir + file_list[i], file_list[i].split(".")[-1])  # 전처리 이미지 저장
            mask.save(MIC5_dir + file_list[i], file_list[i].split(".")[-1])  # 측정결과 이미지 저장

            logging.info(f'Mask saved to {file_list[i]}')

        if args.viz:
            logging.info(f'Visualizing results for image {filename}, close to continue...')
            # plot_img_and_mask(img, mask)

            file_name = filename.split("\\")[-1]
            mask_file_name = file_name.split(".")[0]+"_mask.PNG"

            img_mask = Image.open(".\\data\\masks\\"+mask_file_name)
            img_mask = np.asarray(img_mask)

            img = np.asarray(img)
            # mask = mask[2, :, :]

            mask_back = np.where(mask[0, :, :] == 1, 0, mask[0, :, :])
            mask_true = np.where(mask[1, :, :] == 1, 200, mask[1, :, :])
            mask_false = np.where(mask[2, :, :] == 1, 50, mask[2, :, :])
            mask_tf = np.where(mask[3, :, :] == 1, 255, mask[3, :, :])

            mask = mask_back + mask_true + mask_false + mask_tf
            np.where(mask == 50, 1, mask)
            np.where(mask == 200, 2, mask)
            np.where(mask == 255, 3, mask)
            print(np.unique(mask))



            if np.unique(mask).size > 0:
                fig, arr = plt.subplots(1, 4, figsize=(15, 7))
                arr[0].imshow(img)
                arr[0].set_title(file_list[i])
                ret, thresh = cv2.threshold(img, 70, 255, cv2.THRESH_BINARY)
                arr[1].imshow(img_mask, cmap="gray")
                arr[1].set_title('mask Image')
                arr[2].imshow(mask, cmap="gray")
                arr[2].set_title('Predict Image')
                # data = applyContour(img, mask)
                # arr[2].imshow(prediction[image_index])
                gray = Image.open(filename).convert("L")
                gray = np.asarray(gray)
                constrast = img_Contrast(img)

                # use numpy to convert the pil_image into a numpy array
                numpy_image = np.array(img)

                # convert to a openCV2 image and convert from RGB to BGR format
                opencv_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)

                stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
                disparity = stereo.compute(opencv_image, opencv_image)

                color_coverted = cv2.cvtColor(disparity, cv2.COLOR_BGR2RGB)

                # convert from openCV2 to PIL
                pil_image = Image.fromarray(color_coverted)


                arr[3].imshow(pil_image)
                arr[3].set_title('Predict contour Image')
                plt.show()


