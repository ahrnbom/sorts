"""
Interface for https://github.com/layumi/Person_reID_baseline_pytorch
and https://github.com/michuanhaohao/reid-strong-baseline
"""

import numpy as np
import torch
from pytorch_model_summary import summary
from pathlib import Path
from time import time
import cv2
from math import ceil
from torchvision import transforms

from reid_files.model.ft_ResNet50 import model

import sys
sys.path.append('reid-strong-baseline')
from modeling.baseline import Baseline

from util import smallest_box, vector_similarity, show

def load_model(mode='strong'):
    if mode == 'strong':
        model_path = model_path = Path('reid_files') / 'market_resnet50_model_120_rank1_945.pth'

        if not model_path.is_file():
            print('I AM ERROR! Download the .pth file from https://drive.google.com/open?id=1hn0sXLZ5yJcxtmuY-ItQfYD7hBtHwt7A')
            raise FileNotFoundError(model_path)

        net = Baseline(num_classes=1, last_stride=1, model_path=None,
                   neck='bnneck', neck_feat='lolwat',
                   model_name='resnet50', pretrain_choice='nope')
        net.load_param(model_path)
        net.cuda()
        net.eval()
        
        return net.base, 16
    elif mode == 'person':
        model_path = Path('reid_files') / 'model' / 'ft_ResNet50' / 'net_last.pth'
        
        if not model_path.is_file():
            print('I AM ERROR! Download from https://drive.google.com/open?id=1XVEYb0TN2SbBYOqf8SzazfYZlpH9CxyE')
            raise FileNotFoundError(model_path)
        
        net = model.ft_net(751)
        net.load_state_dict(torch.load(model_path))
        
        def forward(x):
            x = net.model.conv1(x)
            x = net.model.bn1(x)
            x = net.model.relu(x)
            x = net.model.maxpool(x)
            x = net.model.layer1(x)
            x = net.model.layer2(x)
            x = net.model.layer3(x)
            x = net.model.layer4(x)
            return x
            
        return forward, 32
    else:
        raise ValueError(f"Unknown mode: {mode}")

class ReID:
    def __init__(self):
        # self.f is scale factor of the network
        self.model, self.f = load_model()
        
        self.w = 128
        self.h = 256
        
        # Resizing is done in OpenCV to avoid costly conversion to PIL image
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        self.total_vectors = 0
        self.fails_count = 0
        
    def get_vector(self, image, mask):
        self.total_vectors += 1
        
        # Assumes the image is of shape HxWx3, 0-255 range uint8
        
        # First, crop out the part where the object is, avoid unnecessary computations
        box = smallest_box(mask)
        if box is None:
            return None
            
        image = image[box[1]:box[3], box[0]:box[2], :]
        mask = mask[box[1]:box[3], box[0]:box[2]]
        
        # Using OpenCV to rescale the image avoids the costly conversion to PIL image
        scaled = cv2.resize(image, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
        x = self.transform(scaled)
        x = torch.unsqueeze(x, 0)
        x = x.cuda()
        
        # The network downsizes the images by a factor
        mask = cv2.resize(mask.astype(np.uint8), (self.w//self.f, self.h//self.f), 
                          interpolation=cv2.INTER_NEAREST).astype(bool)
        
        y = self.model(x)
        y = y.detach().cpu().numpy()
        y = y[0, :, :, :]  
        
        # If nothing remains of the mask after scaling down (rare, but happens)
        if not np.any(mask):
            #y_masked = y.reshape(-1, y.shape[0]).T
            self.fails_count += 1
            return np.zeros((y.shape[0],), dtype=np.float32), False
        else:
            # Extract the mask part
            y_masked = y[:, mask] 
            success = True
        
        # Masked mean
        vector = np.mean(y_masked, axis=-1)
        
        return vector, success

def main():
    model, _ = load_model()
    x = np.zeros((1,3,256,128), dtype=np.float32)
    x = torch.tensor(x, device='cuda')
    
    n = 64
    before = time()
    for i in range(n):
        y = model(x)
    after = time()
    dt = (after-before)/n
    
    print('Mean execution time:', dt, 's')
    
    r = ReID()
    
    import imageio as iio
    from full_format import decode_one
    
    f5 = iio.imread(Path('MOTS') / 'train' / 'MOTS20-05' / 'img1' / '000005.jpg')
    mask_0_f5 =  decode_one(r'TW?>]>7I7I6H9I7J7H;G=XCXN`;j2J4L5K5K4M4L6K6Ib0_O2M2N1O1O1O100O1O1O1O0KKlFhJR9h4j0102O00100O2O002N1O5Kh0XO1O1O000000000001O00001N1O2]OVGiJP9m4h0K5M3M?A8H3M3L4M3M5J5K5mNgDUNa;d1S1J7I?Bb0]O9D_ci7',
           (480,640))
    
    v0_f5, _ = r.get_vector(f5, mask_0_f5)
    
    mask_1_f5 = decode_one(r'YZR24d><J2M3M3O1N1O2L4K5N3L8PEcNP8R2YGTNb8Z2QGjMm8c2bFbM]9k2iEcMW:c301O0O100O1TOfKQG[4n8kKjFW4V9j000000O001N2K5_Oa0I7M3M3004M2M4L8I4L7J1N1O01N6K6I2N1O1N101N1PNaF]Nb9Y1YGSNi8>jE1o1UOY8f0f3O1O2O00010O01000O100O1O1O2N1O0O2N1O2N2N5KgTU6',
           (480,640))
    
    v1_f5, _ = r.get_vector(f5, mask_1_f5)
    
    f40 = iio.imread(Path('MOTS') / 'train' / 'MOTS20-05' / 'img1' / '000040.jpg')
    
    mask_0_f40 = decode_one(r'fQQ1a0X>=D9F9H7L5lEUNl6o1PIcN^6`1^IeN\6a1^IeN]6_1[IjN_6\1ZIkNa6Y1ZIlNc6X1XImNd6W1WImNe6Y1RIoNg6[1iHQOQ7^1XHPOd7X4M4K4L4L4L5L3N2O2M3N1N3M3M4L6K6I6K8H7J5K9F`0A8H2M3N0O1O2N1N2O1N2O1O1O1O100O10000O010O1O\NbLgG]3V8TM\Gl2a8dMSGR2T9c1;I7N2M4M3M3N2N2O1O1O2O0O102N2M4M2N3M2N3M_1aN2N2N1O1O000000000000000000000001O00000000000O2YNbNjE^1R:nNeES1X:UOcEl0X:]ObEd0\:C^E?`:GZE<c:IWE:g:R2gFRL[7Q4_HUL_7m3[HYLb7j3YH[Le7g3UH^Lj7e3lGcLR8V501O2N2N3L3N2N1O2N1N4M2N4L2M3N4L6J7H7J4L3lMaH`Lf7k2lHQM]7_2mH_M\7P2oHlMW7k1oHRNV7h1mHUNZ7b1lHQNl7[1dgT6',
           (480,640))
    
    v0_f40, _ = r.get_vector(f40, mask_0_f40)
    
    mask_1_f40 = decode_one(r'VSY46g>7G:E9F8\Od0K5J6I7J6J6J6M3L4M5J6I7J6K5K5L5J7I;Ef0YJ[Ji0Z6]NWJW1U6ZNUJ`1S6UNRJi1U6mMPJR2X6`MnI_2[9O0O2O000O1N3N1N2N2N2O1O1O1O1O1O001O1N2M3L4L4L3O2N2O0O100O00N3N1N3H7E<C>lNS1XOh0H9O100O10000O1002N3M9G;E;E<D5K5K7I8H7I5K4L3M4L5K6J4L4M1N3M101O0000O2O1N2N2O2M3_D_M]:d3N2N1O0dGQLf5Q4WJSLg5n3WJTLh5l3XJULf5m3ZJSLd5o3\JSLa5n3^JTLW2^OhN`4RORL^1b0PO`3BoKP1g1YNf2g0mK8]8GkGF`88jGYO^8d0iGSO\8l0iGgN_8X1e2O2N1O1O2N100O1O2N1O1O100O100O2O000O10000O100000000O100O10000O2O000O100000000O2N2N2M3N3L4L4M4K5K6HT]a2',
           (480,640))
    
    v1_f40, _ = r.get_vector(f40, mask_1_f40)
    
    d00 = vector_similarity(v0_f5, v0_f40) # Should be large
    d11 = vector_similarity(v1_f5, v1_f40) # Should be large
    
    d01 = vector_similarity(v0_f5, v1_f5) # Should be small
    d10 = vector_similarity(v0_f40, v1_f40) # Should be small
    
    print(d00, d11, d01, d10)
    
if __name__ == '__main__':
    main()


