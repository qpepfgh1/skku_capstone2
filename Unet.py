#!/usr/bin/env python
# coding: utf-8

# # 라이브러리

# In[140]:


#*---------- basic ------------*
import os, cv2, random, time
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import math
import copy                        # 가중치 복사
import natsort
import json
import time

#*--------- torch ------------*
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

#*--------- torch vision -------*
import torchvision
import torchsummary as summary
import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


#*--------- Learning_rate Scheduler ---------*
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, _LRScheduler

#*-------- warnings ------------*
import warnings
warnings.filterwarnings(action='ignore')


#*---------- Set up random seed ---------*
def set_seed(seed):
    np.random.seed(seed)
    os.environ['PYTHONSHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

set_seed(1010) # randomSeed 고정


# # Custom 라이브러리 

# In[141]:


#*------------- for Unet ++ -----------*
from collections import OrderedDict
from torch.optim import lr_scheduler
from model import Unet_block, UNet, Nested_UNet
from init_weights import init_weights
from utils import BCEDiceLoss, AverageMeter, count_params, iou_score

import jsonMethod
import shutil
import datetime
import image_convertor


# # Configuration

# In[142]:


CFG = {
    'IMG_SIZE' : 224,
    'BATCH_SIZE' : 16,
    'NUM_EPOCHS' : 100,
    'NUM_CLASS' : 4
}

# os.chdir(r'C:\\Users\\SDML\\1_Continue\\4. SmartFactoryCapstoneDesign\\Code\\unet++\\[Final]\\UI')


# # def

# In[143]:


def get_RGB_statis(imgs_paths):  
    
    images = []
    for img_path in imgs_paths:
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.
        images.append(img)
    images = np.array(images)
    
    R_mean = [image[0].mean() for image in images]
    G_mean = [image[1].mean() for image in images]
    B_mean = [image[2].mean() for image in images]
    
    R_std = [image[0].std() for image in images]
    G_std = [image[1].std() for image in images]
    B_std = [image[2].std() for image in images]
    
    R_mean = np.array(R_mean); G_mean = np.array(G_mean); B_mean = np.array(B_mean)
    R_std = np.array(R_std); G_std = np.array(G_std); B_std = np.array(B_std)

    return [R_mean.mean(), G_mean.mean(), B_mean.mean()], [R_std.mean(), G_std.mean(), B_std.mean()]

class CustomDataset(Dataset):
    def __init__(self, true_folder_path, false_folder_path,
                 train_image_Rigid_transform=None,
                 train_image_intensity_transform = None, 
                 train_image_Normalization_trainsform = None,
                 test_transform = None,
                 train = bool):
        
        #image
        self.true_folder_path  = true_folder_path       # 정상 이미지 : 128
        self.false_folder_path = false_folder_path     # 비정상 이미지 : 91
        
        #mask-true
        self.true_mask_background_folder_path  = os.path.join(true_folder_path + '/masks', 'mask_background')
        self.true_mask_Noinfo_folder_path      = os.path.join(true_folder_path + '/masks', 'mask_NoInfo')
        self.true_mask_fault_folder_path       = os.path.join(true_folder_path + '/masks', 'mask_fault')
        self.true_mask_hole_folder_path        = os.path.join(true_folder_path + '/masks', 'mask_hole')
        
        #mask-false
        self.false_mask_background_folder_path  = os.path.join(false_folder_path + '/masks', 'mask_background')
        self.false_mask_Noinfo_folder_path      = os.path.join(false_folder_path + '/masks', 'mask_NoInfo')
        self.false_mask_fault_folder_path       = os.path.join(false_folder_path + '/masks', 'mask_fault')
        self.false_mask_hole_folder_path        = os.path.join(false_folder_path + '/masks', 'mask_hole')

        #transform
        self.train_image_Rigid_transform          = train_image_Rigid_transform
        self.train_image_intensity_transform      = train_image_intensity_transform
        self.train_image_Normalization_trainsform = train_image_Normalization_trainsform
        self.test_transform = test_transform
        
        #train mode
        self.train = train
        
        #ext
        self.extension = '*.PNG'
        
        #true set
        self.true_image_paths           = natsort.natsorted(glob(os.path.join(self.true_folder_path + '/images',  self.extension)))
        self.true_mask_background_paths = natsort.natsorted(glob(os.path.join(self.true_mask_background_folder_path, self.extension)))
        self.true_mask_Noinfo_paths     = natsort.natsorted(glob(os.path.join(self.true_mask_Noinfo_folder_path, self.extension)))
        self.true_mask_fault_paths      = natsort.natsorted(glob(os.path.join(self.true_mask_fault_folder_path, self.extension)))
        self.true_mask_hole_paths       = natsort.natsorted(glob(os.path.join(self.true_mask_hole_folder_path, self.extension)))  

        #false set
        self.false_image_paths           = natsort.natsorted(glob(os.path.join(self.false_folder_path + '/images', self.extension)))
        self.false_mask_background_paths = natsort.natsorted(glob(os.path.join(self.false_mask_background_folder_path, self.extension)))
        self.false_mask_Noinfo_paths     = natsort.natsorted(glob(os.path.join(self.false_mask_Noinfo_folder_path, self.extension)))
        self.false_mask_fault_paths      = natsort.natsorted(glob(os.path.join(self.false_mask_fault_folder_path, self.extension)))
        self.false_mask_hole_paths       = natsort.natsorted(glob(os.path.join(self.false_mask_hole_folder_path, self.extension)))
        
        #organized
        self.true_set = list(zip(self.true_image_paths, self.true_mask_background_paths,
                                 self.true_mask_Noinfo_paths, self.true_mask_fault_paths, self.true_mask_hole_paths))
        self.false_set = list(zip(self.false_image_paths, self.false_mask_background_paths,
                                 self.false_mask_Noinfo_paths, self.false_mask_fault_paths, self.false_mask_hole_paths))
        
        self.result_set = np.concatenate([self.true_set, self.false_set], axis = 0)      
        
    def __getitem__(self, idx):        
        image_path, mask_background_path, mask_Noinfo_path, mask_fault_path, mask_hole_path = self.result_set[idx]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask_background = self.get_mask(mask_background_path, CFG['IMG_SIZE'], CFG['IMG_SIZE'])
        mask_Noinfo     = self.get_mask(mask_Noinfo_path, CFG['IMG_SIZE'], CFG['IMG_SIZE'])
        mask_fault      = self.get_mask(mask_fault_path, CFG['IMG_SIZE'], CFG['IMG_SIZE'])
        mask_hole       = self.get_mask(mask_hole_path, CFG['IMG_SIZE'], CFG['IMG_SIZE'])
        
        mask_background = np.expand_dims(mask_background, axis=-1)
        mask_Noinfo     = np.expand_dims(mask_Noinfo, axis=-1)
        mask_fault      = np.expand_dims(mask_fault, axis=-1)
        mask_hole       = np.expand_dims(mask_hole, axis=-1)
        
        if self.train:
            #Rigid Tranformation : images, masks
            if self.train_image_Rigid_transform is not None:
                Rigid_transform                      = self.train_image_Rigid_transform(image = image,
                                                                                        mask_background = mask_background,
                                                                                        mask_Noinfo = mask_Noinfo,
                                                                                        mask_fault = mask_fault,
                                                                                        mask_hole = mask_hole)

                image           = Rigid_transform['image']
                mask_background = Rigid_transform['mask_background']
                mask_Noinfo     = Rigid_transform['mask_Noinfo']
                mask_fault      = Rigid_transform['mask_fault']
                mask_hole       = Rigid_transform['mask_hole']
            #intensity Tranformation : images
            if self.train_image_intensity_transform is not None:
                train_image_intensity_transform      = self.train_image_intensity_transform(image = image)
                image = train_image_intensity_transform['image']
            #Normalization Tranformation : images
            train_image_Normalization_trainsform = self.train_image_Normalization_trainsform(image = image)
            image = train_image_Normalization_trainsform['image']

            mask_background = torch.Tensor(mask_background)
            mask_Noinfo     = torch.Tensor(mask_Noinfo)
            mask_fault      = torch.Tensor(mask_fault)
            mask_hole       = torch.Tensor(mask_hole)

            mask_background = torch.permute(mask_background, (2,0,1))
            mask_Noinfo = torch.permute(mask_Noinfo, (2,0,1))
            mask_fault = torch.permute(mask_fault, (2,0,1))
            mask_hole = torch.permute(mask_hole, (2,0,1))
            
            result_mask = np.concatenate([mask_background, mask_Noinfo, mask_fault, mask_hole], axis = 0)
            
            return image, result_mask
        else:
            test_transform = self.test_transform(image = image)
            image = test_transform['image']
            
            mask_background = torch.Tensor(mask_background)
            mask_Noinfo     = torch.Tensor(mask_Noinfo)
            mask_fault      = torch.Tensor(mask_fault)
            mask_hole       = torch.Tensor(mask_hole)
            
            mask_background = torch.permute(mask_background, (2,0,1))
            mask_Noinfo = torch.permute(mask_Noinfo, (2,0,1))
            mask_fault = torch.permute(mask_fault, (2,0,1))
            mask_hole = torch.permute(mask_hole, (2,0,1))
            result_mask = np.concatenate([mask_background, mask_Noinfo, mask_fault, mask_hole], axis = 0)
            return image, result_mask
        
    def __len__(self):
        return len(self.result_set)

    
    def get_mask(self, mask_path, IMG_HEIGHT, IMG_WIDTH):
        mask = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)   
        mask_ = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask_ = cv2.resize(mask_, (IMG_HEIGHT, IMG_WIDTH))
        mask += mask_  
        # 들어오는 마스크의 픽셀은 0과 1로 이루어짐. sigmoid를 위한 처리는 필요없음.
        # mask =  mask/ 255.
        return mask

class CosineAnnealingWarmUpRestarts(_LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, eta_max=0.1, T_up=0, gamma=1., last_epoch=-1):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        if T_up < 0 or not isinstance(T_up, int):
            raise ValueError("Expected positive integer T_up, but got {}".format(T_up))
        self.T_0 = T_0
        self.T_mult = T_mult
        self.base_eta_max = eta_max
        self.eta_max = eta_max
        self.T_up = T_up
        self.T_i = T_0
        self.gamma = gamma
        self.cycle = 0
        self.T_cur = last_epoch
        super(CosineAnnealingWarmUpRestarts, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.T_cur == -1:
            return self.base_lrs
        elif self.T_cur < self.T_up:
            return [(self.eta_max - base_lr)*self.T_cur / self.T_up + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.eta_max - base_lr) * (1 + math.cos(math.pi * (self.T_cur-self.T_up) / (self.T_i - self.T_up))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.cycle += 1
                self.T_cur = self.T_cur - self.T_i
                self.T_i = (self.T_i - self.T_up) * self.T_mult + self.T_up
        else:
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                    self.cycle = epoch // self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.cycle = n
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
                
        self.eta_max = self.base_eta_max * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr
            
#set up
def train(train_loader, model, criterion, optimizer, device):
    avg_meters = {'loss':AverageMeter(),
                  'iou' :AverageMeter()}

    model.train()
    # pbar = tqdm(total=len(train_loader))

    for inputs, labels in train_loader:
        
        inputs = torch.tensor(inputs, device=device, dtype=torch.float32)
        labels = torch.tensor(labels, device=device, dtype=torch.float32)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # log
        iou = iou_score(outputs, labels, threshold=0.8)
        avg_meters['loss'].update(loss.item(), n=inputs.size(0))
        avg_meters['iou'].update(iou, n=inputs.size(0))

        log = OrderedDict([
                    ('loss', avg_meters['loss'].avg),
                    ('iou', avg_meters['iou'].avg),
                ])
    return log, model

#set-up
def validation(val_loader, model, criterion, device):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter()}
    
    model.eval()
    with torch.no_grad():
#        pbar = tqdm(total=len(val_loader))
        for inputs, labels in val_loader:
            inputs = torch.tensor(inputs, device=device, dtype=torch.float32)
            labels = torch.tensor(labels, device=device, dtype=torch.float32)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            iou = iou_score(outputs, labels, threshold=0.8)

            avg_meters['loss'].update(loss.item(), n=inputs.size(0))
            avg_meters['iou'].update(iou, n=inputs.size(0))

            log = OrderedDict([
                        ('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg),
                    ])
    return log

def TRAIN(Model_name='Sample_Model', # 폴더 생성 이름
          num_class=4, # 클래스
          trainData_path = './src', # 데이터셋 저장 경로, train, valid 데이터 찾아가서 dataloader 만드는 형식
          Batch_size = 8, # 배치사이즈
          Epochs = 30, # 러닝레이트
          LR = 0.00068, # 에포크
          Deep_supervision = False): # 딥 수퍼비전 사용 유무

    # 학습 시작하면 json 파일에 학습 과정 저장되는 형태 (에포크 별로 기록 x)
    # TRAIN(Model_name, num_class, trainData_path, Batch_size, Epochs, LR, Deep_supervision)

    if not os.path.isdir('./src/model'):
        os.mkdir('./src/model')
        
    if not os.path.isdir('./src/model/%s'%(Model_name)):
        os.mkdir('./src/model/%s'%(Model_name))
        os.mkdir('./src/model/%s/result'%(Model_name))
        os.mkdir('./src/model/%s/result/weights'%(Model_name))
    
    ############ get statistics from Train Dataset ############
    ############# 학습 데이터에 대한 통계값 추출 ##############
    
    # 학습 데이터 경로
    true_path = os.path.join(trainData_path, 'train/true_v3/images')
    false_path = os.path.join(trainData_path, 'train/false_v7/images')

    true_paths = glob(os.path.join(true_path, '*.PNG'))
    false_paths = glob(os.path.join(false_path, '*.PNG'))

    #concat
    true_paths = np.array(true_paths); false_paths = np.array(false_paths)
    imgs_paths = np.concatenate([true_paths, false_paths], axis = 0)

    RGB_mean, RGB_std = get_RGB_statis(imgs_paths)
    ############################################################
    
    ############ transform ############
    train_image_Rigid_transform = A.Compose([
                                A.Resize(CFG['IMG_SIZE'],CFG['IMG_SIZE'])])
    train_image_intensity_transform = A.Compose([
                                A.GaussNoise(p=0.5),
                                A.OneOf([
                                    A.RandomBrightnessContrast(p=0.5),
                                    A.RandomContrast(p=0.5),
                                    A.RandomGamma(p=0.5),
                                    A.JpegCompression(p=0.5),
                                    A.CLAHE(clip_limit=10.0, tile_grid_size=(10, 10), p=0.5)
                                ], p=1.0)])                               
    train_image_Normalization_trainsform = A.Compose([
                                A.Normalize(mean= (RGB_mean[0], RGB_mean[1], RGB_mean[2]),
                                            std= (RGB_std[0], RGB_std[1], RGB_std[2]), max_pixel_value=255.0, always_apply=False, p=1.0),
                                ToTensorV2()])
    
    test_transform = A.Compose([
                                A.Resize(CFG['IMG_SIZE'],CFG['IMG_SIZE']),
                                A.Normalize(mean= (RGB_mean[0]),
                                            std= (RGB_std[0]), max_pixel_value=255.0, always_apply=False, p=1.0),
                                ToTensorV2()])
    ####################################
    
    ############ create Dataloader #############
    train_true_folder_path = os.path.join(trainData_path, 'train/true_v3')
    train_false_folder_path = os.path.join(trainData_path, 'train/false_v7')
    valid_true_folder_path = os.path.join(trainData_path, 'valid/true_v3')
    valid_false_folder_path = os.path.join(trainData_path, 'valid/false_v7')

    trainDataset = CustomDataset(train_true_folder_path, train_false_folder_path,
                                 train_image_Rigid_transform, train_image_intensity_transform, train_image_Normalization_trainsform,
                                 train = True)


    validDataset = CustomDataset(valid_true_folder_path, valid_false_folder_path,
                                 train_image_Rigid_transform, train_image_intensity_transform, train_image_Normalization_trainsform,
                                 test_transform, train = False)

    Train_Dataloader = DataLoader(trainDataset, batch_size = Batch_size, shuffle = True)
    Valid_Dataloader = DataLoader(validDataset, batch_size = Batch_size, shuffle = False)   
    
    ############ Train by CustomDataset ############
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # best 값 미리 int 형식으로 선언
    best_Diceloss = 10
    best_iou = 0
    
    #model, loss, optimizer 선언
    model = Nested_UNet(num_classes = num_class, input_channels = 3, deep_supervision= Deep_supervision).to(device)
    criterion = BCEDiceLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LR)
    
    #ConsineAnnealing
    lr_scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=100, T_mult=2, eta_max=1e-5,  T_up=10, gamma=0.5)
    
    #train_start
    start = time.time()
    for epoch in range(1, Epochs+1):

        jsonData = {}
        jsonData['total'] = Epochs
        jsonData['count'] = epoch
        jsonMethod.setData(type="TRAIN", data=jsonData)
        
        #학습
        train_log, model = train(Train_Dataloader, model, criterion, optimizer, device)
        #검증
        val_log =  validation(Valid_Dataloader, model, criterion, device)

        end = time.time()

        #cosine Annealing learning rate scheduler, 한스텝 이동
        lr_scheduler.step()
        
        print(f'{epoch}Epoch')
        print(f'train Dice loss:{train_log["loss"]:.3f} |train iou:{train_log["iou"]:.3f}')
        print(f'val Dice loss:{val_log["loss"]:.3f} |val iou:{val_log["iou"]:.3f}\n')
        
        #best 결과 저장
        if val_log['loss'] < best_Diceloss:
            best_iou = val_log['iou']
            best_Diceloss = val_log['loss']
            best_epoch = epoch
            best_weights = copy.deepcopy(model.state_dict())
            
            #베스트 가중치 저장
            torch.save(best_weights, './src/model/'+Model_name+'.pth')
            
        print(f'Best iou : {best_iou:.3f} Best Dice loss : {best_Diceloss:.3f}')
        print('-'*30)
        
        ########################## 학습 과정 출력 #############################
        Meta_data = {
            Model_name:{
                'current_epochs': str(epoch),
                'End_epochs': str(Epochs),
                'time': f'{end - start:.2f}',
                'train_dice_loss': f'{train_log["loss"]:.3f}',
                'train_iou': f'{train_log["iou"]:.3f}',
                'val_dice_loss' : f'{val_log["loss"]:.3f}',
                'val_iou' : f'{val_log["iou"]:.3f}',
                'Best_iou': f'{best_iou:.3f}',
                'Best_dice_loss': f'{best_Diceloss:.3f}'}
        }
        #######################################################################
            
        # 학습 과정 저장
        with open('./src/model/%s/result/result.json'%(Model_name), "w") as f:
            json.dump(Meta_data, f)
    
    #best model update and return
    best_weights_path = './src/model/'+Model_name+'.pth'
    checkpoint = torch.load(best_weights_path)
    best_model = model.load_state_dict(checkpoint)

    end = time.time()
    sec = (end - start)
    result = datetime.timedelta(seconds=sec)
    result_list = str(datetime.timedelta(seconds=sec)).split(".")
    print(result_list[0])

    json_data = {}
    with open(os.path.join("./src/model/", 'result.json'), 'r', encoding="UTF-8") as f:
        json_data = json.load(f)
    print(json_data)

    json_data[Model_name + ".pth"] = {}
    json_data[Model_name + ".pth"]["time"] = str(result_list[0])
    json_data[Model_name + ".pth"]["num"] = str(Epochs)
    json_data[Model_name + ".pth"]["acc"] = "{}".format(best_iou)
    json_data[Model_name + ".pth"]["loss"] = "{}".format(best_Diceloss)
    with open(os.path.join("./src/model/", 'result.json'), 'w', encoding="utf-8") as make_file:
        json.dump(json_data, make_file, ensure_ascii=False, indent="\t")

    return best_model, Meta_data


# In[135]:





# # Train Result Visualization

# In[144]:


class TestDataset(Dataset):
    def __init__(self, test_path, 
                 test_transform = None):
        
        self.test_transform = test_transform
        self.test_path = test_path
        self.test_paths = glob(test_path + '/' + '*.PNG')
        
    def __getitem__(self, idx):        
        image = cv2.imread(self.test_paths[idx], cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        #Normalization Tranformation : images
        test_transform = self.test_transform(image = image)
        image = test_transform['image']

        return image
        
    def __len__(self):
        return len(self.test_paths)


def Test(Model_name = 'Sample_Model',           # 학습했던 모델 경로
        num_class = 4,                        # 클래스
        Deep_supervision = False,            # 학습했던 모델의 딥 수퍼비전 사용 유무
        trainData_path = './src',             # 학습 시 사용됐던 통계값 추출을 위한 train 데이터 사용
        TestData_path = './src/test/images'):  # 통계값을 기준으로 테스트셋에 대하여 정규화 진행하고 추론 진행):
    # if not os.path.isdir('./src/model/%s/result/test_result'%(Model_name)):
    #     os.mkdir('./src/model/%s/result/test_result'%(Model_name))
    #
    # for idx in range(2):
    #     result_save_path = './src/model/%s/result/test_result/%d'%(Model_name, idx)
    #     if not os.path.isdir(result_save_path):
    #         os.mkdir(result_save_path)

    weights_path = './src/model/'+Model_name

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = Nested_UNet(num_classes = 4, input_channels = 3, deep_supervision= Deep_supervision).to(device)
    check_point = torch.load(weights_path)
    model.load_state_dict(check_point)
    Best_model = model.to(device)

    ############ get statistics from Train Dataset ############
    true_path = os.path.join(trainData_path, 'train/true_v3/images')
    false_path = os.path.join(trainData_path, 'train/false_v7/images')

    true_paths = glob(os.path.join(true_path, '*.PNG'))
    false_paths = glob(os.path.join(false_path, '*.PNG'))

    #concat
    true_paths = np.array(true_paths); false_paths = np.array(false_paths)
    imgs_paths = np.concatenate([true_paths, false_paths], axis = 0)

    RGB_mean, RGB_std = get_RGB_statis(imgs_paths)
    ############################################################
    test_transform = A.Compose([
                                A.Resize(CFG['IMG_SIZE'],CFG['IMG_SIZE']),
                                A.Normalize(mean= (RGB_mean[0], RGB_mean[1], RGB_mean[2]),
                                            std= (RGB_std[0], RGB_std[1], RGB_std[2]), max_pixel_value=255.0, always_apply=False, p=1.0),
                                ToTensorV2()])


    testDataset = TestDataset(TestData_path, test_transform)

    Test_Dataloader = DataLoader(testDataset, batch_size = 1, shuffle = False)

    Best_model.eval()
    
    total_inference_time = 0
    Testimg_paths = glob(TestData_path + './' + '*.PNG')
    for idx, (image, img_path) in enumerate(zip(Test_Dataloader, Testimg_paths)):
        start = time.time()
        image = image.to(device);

        #mask prediction
        seg_image = Best_model(image)
        seg_image = torch.sigmoid(seg_image)

        seg_image[seg_image<0.5]=0
        seg_image[seg_image>=0.5]=1
        seg_image = seg_image.squeeze(dim = 0)

        mapping = lambda x:x*255

        prediction_mask_background = mapping(seg_image.permute(1, 2, 0)[:, :, 0].detach().cpu().numpy().astype(np.uint8))
        prediction_mask_NoInfo     = mapping(seg_image.permute(1, 2, 0)[:, :, 1].detach().cpu().numpy().astype(np.uint8))
        prediction_mask_fault      = mapping(seg_image.permute(1, 2, 0)[:, :, 2].detach().cpu().numpy().astype(np.uint8))
        prediction_mask_hole       = mapping(seg_image.permute(1, 2, 0)[:, :, 3].detach().cpu().numpy().astype(np.uint8))

        end = time.time()
        total_inference_time += end - start

        ########################## 학습 과정 출력 #############################
        test_info = {
            Model_name:{
                'inferenced_numOfTestData': str(idx + 1),
                'numOfTestData': str(len(testDataset)),
                'inference_time': f'{end - start:.2f}',
                'total_inference_time' : f'{total_inference_time:.2f}'
        }}
        #######################################################################

        #test result save
        # with open('./src/result/%s/result.json'%(Model_name), "w") as f:
        #     json.dump(test_info, f)

        #원본 이미지
        getfilename = img_path.split('\\')[-1]
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (224,224))
        
        '''0폴더에는 학습 이미지 저장'''
        cv2.imwrite("./src/result/" + Model_name + "/img/MIC1/"+getfilename, img)

        # 이미지 blending
        # 파란색 -> 홀, 빨간색 -> 결함, 초록색 -> noinfo
        zeros = np.zeros((224,224, 1))
        masks = np.concatenate([zeros,prediction_mask_hole[:,:,np.newaxis],prediction_mask_fault[:,:,np.newaxis]], axis = 2)
        blend_img = cv2.add(img, masks)
        '''1폴더에는 추론 결과 이미지 저장'''
        cv2.imwrite("./src/result/" + Model_name + "/img/MIC4/"+getfilename, blend_img)
        cv2.imwrite("./src/result/" + Model_name + "/img/MIC5/"+getfilename, blend_img)


if __name__ == '__main__':
    TRAIN()
# In[146]:


# Model_name = 'Sample_Model'           # 학습했던 모델 경로
# num_class = 4                        # 클래스
# Deep_supervision = False            # 학습했던 모델의 딥 수퍼비전 사용 유무
# trainData_path = './src'             # 학습 시 사용됐던 통계값 추출을 위한 train 데이터 사용
# TestData_path = './src/test/images'  # 통계값을 기준으로 테스트셋에 대하여 정규화 진행하고 추론 진행
# Test(Model_name, num_class, Deep_supervision, trainData_path, TestData_path) # 추론
#
#
# # In[153]:
#
#
# if not os.path.isdir('./src/model/%s/result/test_result'%(Model_name)):
#     os.mkdir('./src/model/%s/result/test_result'%(Model_name))
#
# for idx in range(2):
#     result_save_path = './src/model/%s/result/test_result/%d'%(Model_name, idx)
#     if not os.path.isdir(result_save_path):
#         os.mkdir(result_save_path)
#
# weights_path = './src/model/%s/result/weights/Best.pt'%(Model_name)
# MetaData_path = './src/model/%s/result/result.json'%(Model_name)
#
# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# model = Nested_UNet(num_classes = 4, input_channels = 3, deep_supervision= Deep_supervision).to(device)
# check_point = torch.load(weights_path)
# model.load_state_dict(check_point)
# Best_model = model.to(device)
#
# ############ get statistics from Train Dataset ############
# true_path = os.path.join(trainData_path, 'train/true_v3/images')
# false_path = os.path.join(trainData_path, 'train/false_v7/images')
#
# true_paths = glob(os.path.join(true_path, '*.PNG'))
# false_paths = glob(os.path.join(false_path, '*.PNG'))
#
# #concat
# true_paths = np.array(true_paths); false_paths = np.array(false_paths)
# imgs_paths = np.concatenate([true_paths, false_paths], axis = 0)
#
# RGB_mean, RGB_std = get_RGB_statis(imgs_paths)
# ############################################################
# test_transform = A.Compose([
#                             A.Resize(CFG['IMG_SIZE'],CFG['IMG_SIZE']),
#                             A.Normalize(mean= (RGB_mean[0], RGB_mean[1], RGB_mean[2]),
#                                         std= (RGB_std[0], RGB_std[1], RGB_std[2]), max_pixel_value=255.0, always_apply=False, p=1.0),
#                             ToTensorV2()])
#
# with open(MetaData_path, 'r') as f:
#     record = json.load(f)
#
#
# testDataset = TestDataset(TestData_path, test_transform)
#
# Test_Dataloader = DataLoader(testDataset, batch_size = 1, shuffle = False)
#
# Best_model.eval()
#
# total_inference_time = 0
# Testimg_paths = glob(TestData_path + './' + '*.PNG')
# for idx, (image, img_path) in enumerate(zip(Test_Dataloader, Testimg_paths)):
#     start = time.time()
#     image = image.to(device);
#
#     #mask prediction
#     seg_image = Best_model(image)
#     seg_image = torch.sigmoid(seg_image)
#
#     seg_image[seg_image<0.5]=0
#     seg_image[seg_image>=0.5]=1
#     seg_image = seg_image.squeeze(dim = 0)
#
#     mapping = lambda x:x*255
#
#     prediction_mask_background = mapping(seg_image.permute(1, 2, 0)[:, :, 0].detach().cpu().numpy().astype(np.uint8))
#     prediction_mask_NoInfo     = mapping(seg_image.permute(1, 2, 0)[:, :, 1].detach().cpu().numpy().astype(np.uint8))
#     prediction_mask_fault      = mapping(seg_image.permute(1, 2, 0)[:, :, 2].detach().cpu().numpy().astype(np.uint8))
#     prediction_mask_hole       = mapping(seg_image.permute(1, 2, 0)[:, :, 3].detach().cpu().numpy().astype(np.uint8))
#
#     end = time.time()
#     total_inference_time += end - start
#
#     ########################## 학습 과정 출력 #############################
#     test_info = {
#         Model_name:{
#             'inferenced_numOfTestData': str(idx + 1),
#             'numOfTestData': str(len(testDataset)),
#             'inference_time': f'{end - start:.2f}',
#             'total_inference_time' : f'{total_inference_time:.2f}'
#     }}
#     #######################################################################
#
#     #test result save
#     with open('./src/model/%s/result/test_result/test_info.json'%(Model_name), "w") as f:
#         json.dump(test_info, f)
#
#     #원본 이미지
#     getfilename = img_path.split('\\')[-1]
#     img = cv2.imread(img_path, cv2.IMREAD_COLOR)
#     img = cv2.resize(img, (224,224))
#
#     '''0폴더에는 학습 이미지 저장'''
#     cv2.imwrite('./src/model/%s/result/test_result/0/'%(Model_name)+getfilename, img)
#
#     # 이미지 blending
#     # 파란색 -> 홀, 빨간색 -> 결함, 초록색 -> noinfo
#     zeros = np.zeros((224,224,1))
#     masks = np.concatenate([zeros,prediction_mask_hole[:,:,np.newaxis],prediction_mask_fault[:,:,np.newaxis]], axis = 2)
#     blend_img = cv2.add(img, masks, dtype=cv2.CV_64F)
#     '''1폴더에는 추론 결과 이미지 저장'''
#     cv2.imwrite('./src/model/%s/result/test_result/1/'%(Model_name)+getfilename, blend_img)

