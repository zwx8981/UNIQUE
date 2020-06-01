import torch
from torchvision import transforms
from BaseCNN import BaseCNN
from PIL import Image
import os
import scipy.io as sio
import pandas as pd
from Main import parse_config
from tqdm import tqdm
from DBCNN import DBCNN
from Transformers import AdaptiveResize

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

test_transform = transforms.Compose([
    AdaptiveResize(512),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225))
])

config = parse_config()
config.backbone = 'resnet34'
config.representation = 'BCNN'
model = DBCNN(config)
model = torch.nn.DataParallel(model).cuda()
# model = BaseCNN2(config)
# model = torch.nn.DataParallel(model).cuda()
model_name = type(model).__name__
print(model)
#[3, 7, 3, 3, 3, 3, 5, 7, 5, 3]
#dbcnn [3, 0, 3, 1, 6, 0, 3, 4, 4, 3]
ckpt = '/home/redpanda/codebase/kede_icip/checkpoint/7/DataParallel-00006.pt'
#ckpt = '/home/redpanda/baidunetdiskdownload/DataParallel-00004.pt'
#ckpt = '/media/redpanda/data/final_version/checkpoint/1/DataParallel-00006.pt'
#ckpt = '/media/redpanda/data/final_version/checkpoint/3/DataParallel-00006.pt'
checkpoint = torch.load(ckpt)
model.load_state_dict(checkpoint['state_dict'])

model.to(device)
model.eval()

compute_waterloo = True

if compute_waterloo:
    waterloo = '/media/redpanda/data/exploration_database_and_code/exploration_database_and_code/distorted_images'
    waterloo_pristine = '/media/redpanda/data/exploration_database_and_code/exploration_database_and_code/pristine_images'
    scores = torch.zeros(99624, 1)
    stds = torch.zeros(99624, 1)

    idx = 0

    pristine_names = os.listdir(waterloo_pristine)
    pristine_names.sort()

    for p, pfile in enumerate(tqdm(pristine_names)):
        I = Image.open(os.path.join(waterloo_pristine, pfile))
        I = test_transform(I)
        I = torch.unsqueeze(I, dim=0)
        I = I.to(device)
        with torch.no_grad():
            score, std = model(I)
            # score = torch.sigmoid(score)
        score = score.cpu().item()
        std = std.cpu().item()
        scores[idx, 0] = score
        stds[idx, 0] = std
        idx = idx + 1
        for i in range(4):
            for j in range(5):
                image_root = os.path.join(waterloo, str(i + 1), str(j + 1))
                I = Image.open(os.path.join(image_root, pfile))
                I = test_transform(I)

                I = torch.unsqueeze(I, dim=0)
                I = I.to(device)
                with torch.no_grad():
                    score, std = model(I)
                    # score = torch.sigmoid(score)
                score = score.cpu().item()
                std = std.cpu().item()
                scores[idx, 0] = score
                stds[idx, 0] = std
                idx = idx + 1

    scores = scores.numpy()
    stds = stds.numpy()
    gMAD_path = '/home/zwx-sjtu/codebase/gMADToolboxV1.0-beta3/gmad_scores.mat'
    sio.savemat(gMAD_path, {'gmad': scores, 'gmad_std': stds})

csv_file = '/home/zwx-sjtu/codebase/gMADToolboxV1.0-beta3/live_path.txt'
data = pd.read_csv(csv_file, sep='\t', header=None)
live_scores = torch.zeros((982, 1))
live_stds = torch.zeros((982, 1))

compute_live = True

if compute_live:
    for i in tqdm(range(982)):
        imgpath = data.iloc[i, 0]
        I = Image.open(imgpath)
        I = test_transform(I)
        I = torch.unsqueeze(I, dim=0)
        I = I.to(device)
        with torch.no_grad():
            score, std = model(I)
            # score = torch.sigmoid(score)
        score = score.cpu().item()
        std = std.cpu().item()
        live_scores[i] = score
        live_stds[i] = std

    scores = live_scores.numpy()
    stds = live_stds.numpy()
    gMAD_path = '/home/zwx-sjtu/codebase/gMADToolboxV1.0-beta3/live_scores.mat'
    sio.savemat(gMAD_path, {'live': scores, 'live_stds': live_stds})

compute_coco = False

if compute_coco:
    coco = '/media/redpanda/data/train2017'
    csv_file = '/media/redpanda/data/coco.txt'
    data = pd.read_csv(csv_file, header=None)
    coco_scores = torch.zeros(118060, 1)
    coco_stds = torch.zeros(118060, 1)

    idx = 0

    for i in tqdm(range(118060)):
        imgpath = data.iloc[i, 0]
        I = Image.open(imgpath)
        I = test_transform(I)
        I = torch.unsqueeze(I, dim=0)
        I = I.to(device)
        with torch.no_grad():
            score, std = model(I)
            # score = torch.sigmoid(score)
        score = score.cpu().item()
        std = std.cpu().item()
        coco_scores[i] = score
        coco_stds[i] = std

    scores = coco_scores.numpy()
    stds = coco_stds.numpy()
    gMAD_path = '/home/zwx-sjtu/codebase/gMADToolboxV1.0-beta3/real_scores.mat'
    sio.savemat(gMAD_path, {'realistic': scores, 'realistic_std': stds})



compute_spaq = True

if compute_spaq:
    spaq = '/media/redpanda/data/SPAQ_database'
    csv_file = '/media/redpanda/data/spaq.txt'
    data = pd.read_csv(csv_file, header=None)
    spaq_scores = torch.zeros(11125, 1)
    spaq_stds = torch.zeros(11125, 1)

    idx = 0

    for i in tqdm(range(11125)):
        imgpath = data.iloc[i, 0]
        I = Image.open(imgpath)
        I = test_transform(I)
        I = torch.unsqueeze(I, dim=0)
        I = I.to(device)
        with torch.no_grad():
            score, std = model(I)
            # score = torch.sigmoid(score)
        score = score.cpu().item()
        std = std.cpu().item()
        spaq_scores[i] = score
        spaq_stds[i] = std

    scores = spaq_scores.numpy()
    stds = spaq_stds.numpy()
    gMAD_path = '/home/zwx-sjtu/codebase/gMADToolboxV1.0-beta3/spaq_scores_dbcnn.mat'
    sio.savemat(gMAD_path, {'spaq': scores, 'spaq_std': stds})