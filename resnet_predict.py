import json

import numpy as np
import torch
import torchvision
from PIL import Image
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import transforms

imagenet_class_idx = json.load(open('data/imagenet_class_index.json'))
model = torchvision.models.resnet101(pretrained=True)
model.eval()
use_gpu = torch.cuda.is_available()
if use_gpu:
    model = model.cuda()

tensor_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224)
])


def pil_to_tensor(pil, norms=([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), use_gpu=torch.cuda.is_available()):
    x = tensor_transforms(pil)
    x = np.asarray(x, dtype=np.float32)
    x = x.transpose((2, 0, 1))
    tensor = torch.from_numpy(x)
    if use_gpu:
        tensor = tensor.cuda()

    tensor = tensor / 255
    mean = norms[0]
    std = norms[1]
    ''' neet to test whether accelerated on GPU '''
    for t, m, s in zip(tensor, mean, std):
        t.sub_(m).div_(s)

    return tensor


def predict(img):
    assert model.training == False, 'model not in eval'
    x = pil_to_tensor(img)
    x = Variable(torch.unsqueeze(x, 0))
    outputs = F.softmax(model(x), dim=1)
    scores, idx = torch.topk(outputs, 5)

    scores = scores.cpu().data.numpy()[0]
    idx = idx.cpu().data.numpy()[0]
    result = []
    for i, score in zip(idx, scores):
        tag = imagenet_class_idx[str(i)][1]
        result.append({'class': tag, 'score': float(score)})

    return result


# initialize before first http request
img = Image.open('data/lenna.jpg').convert('RGB')
predict(img)
