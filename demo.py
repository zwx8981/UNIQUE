import torch
from torchvision import transforms
from BaseCNN import BaseCNN
from Main import parse_config
from Transformers import AdaptiveResize
from PIL import Image

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

test_transform = transforms.Compose([
    AdaptiveResize(768),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225))
])

config = parse_config()
config.backbone = 'resnet34'
config.representation = 'BCNN'

model = BaseCNN(config)
model = torch.nn.DataParallel(model).cuda()

ckpt = './model.pt'

checkpoint = torch.load(ckpt)
model.load_state_dict(checkpoint)

image1 = './demo/test1.JPG'
image2 = './demo/test2.png'
image3 = './demo/test3.bmp'

image1 = Image.open(image1)
image1 = test_transform(image1)
image1 = torch.unsqueeze(image1, dim=0)
image1 = image1.to(device)
with torch.no_grad():
    score1, std1 = model(image1)

score1 = score1.cpu().item()
std1 = std1.cpu().item()
print('The predicted quality of image1 is {}, with an estimated std of {}'.format(score1, std1))

image2 = Image.open(image2)
image2 = test_transform(image2)
image2 = torch.unsqueeze(image2, dim=0)
image2 = image2.to(device)
with torch.no_grad():
    score2, std2 = model(image2)

score2 = score2.cpu().item()
std2 = std2.cpu().item()
print('The predicted quality of image2 is {}, with an estimated std of {}'.format(score2, std2))

image3 = Image.open(image3)
image3 = test_transform(image3)
image3 = torch.unsqueeze(image3, dim=0)
image3 = image3.to(device)
with torch.no_grad():
    score3, std3 = model(image3)

score3 = score3.cpu().item()
std3 = std3.cpu().item()
print('The predicted quality of image3 is {}, with an estimated std of {}'.format(score3, std3))