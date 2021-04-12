import os
import cv2
import torch
import random
import numpy as np

from model import Generator
class video2anime:
    def __init__(self, device, size):
        self.device = device
        self.image_size = size # Can be tuned, works best when the face width is between 200~250 px
        self.model = Generator().eval().to(self.device)
        torch.set_grad_enabled(False)
        self.ckpt = torch.load(f"checkpoint/generator_celeba_distill.pt", map_location=self.device)
        self.model.load_state_dict(self.ckpt)

    def load_image(self, image, size=None):
        image = self.image2tensor(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        w, h = image.shape[-2:]
        if w != h:
            crop_size = min(w, h)
            left = (w - crop_size)//2
            right = left + crop_size
            top = (h - crop_size)//2
            bottom = top + crop_size
            image = image[:,:,left:right, top:bottom]

        if size is not None and image.shape[-1] != size:
            image = torch.nn.functional.interpolate(image, (size, size), mode="bilinear", align_corners=True)
        
        return image

    def image2tensor(self, image):
        image = torch.FloatTensor(image).permute(2,0,1).unsqueeze(0)/255.
        return (image-0.5)/0.5

    def tensor2image(self, tensor):
        tensor = tensor.clamp_(-1., 1.).detach().squeeze().permute(1,2,0).cpu().numpy()
        return tensor*0.5 + 0.5

    def style_transfer(self, image):
        image = self.load_image(image, self.image_size)
        output = self.model(image.to(self.device))
        output = cv2.cvtColor(255*self.tensor2image(output), cv2.COLOR_BGR2RGB)
        return output
