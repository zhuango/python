from torchvision import models

print(dir(models))
alexnet = models.AlexNet()
renet = models.resnet101(pretrained=True)

from PIL import Image
img = Image.open("./dog.PNG")
print(img)
img.show()
img_t = preprocess(img)

import torch
# add batch dimension.
batch_t = torch.unsequence(img_t, 0)
resnet.eval()
out = resnet(batch_t)
print(out)