import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
from torchvision import transforms

# ref: https://blog.csdn.net/lyl771857509/article/details/84175874
class VGG(nn.Module):
    def __init__(self, weights=False):
        super(VGG, self).__init__()
        if weights is False:
            self.model = models.vgg19(pretrained=True)
        else:
            self.model = models.vgg19(pretrained=False)
            pre = torch.load(weights)
            self.model.load_state_dict(pre)
        # output features not predictions.
        self.vgg19 = self.model.features
        
        for param in self.vgg19.parameters():
            param.requires_grad = False
    def __call__(self, input):
        return self.vgg19(input)

tran=transforms.Compose([
    # fixed image size for VGG.
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
])

# Load images.
img = Image.open("./CXR3498_IM-1702-1001.png")
im=tran(img)
im.unsqueeze_(dim=0)
print(im.shape)

model = VGG(False)
res = model(im)
print(res.size())