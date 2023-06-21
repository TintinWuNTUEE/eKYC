import os
from glob import glob

import torch
from torchvision import transforms as T
from facenet_pytorch import MTCNN

from .model.model import get_models
from PIL import Image
from io import BytesIO
import base64


class eKYC_PAD:
    def __init__(self,mtcnn) :
        '''
        init
        '''
        self.__device = self.__get_device()
        self.__path = '/home/tintinw/eKYC/backend/controller/APP/PAD/weights_efficient'
        self.__mtcnn = mtcnn
        self.__model = get_models()
        self.__model = self.__load_model(self.__model,self.__path)
        self.__transform = self.__get_transform()
    def __get_device(self):
        device = None
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        return device
    def __load_model(self,model,path):
        '''
        Load model
        '''
        if os.listdir(path):
            file_path = sorted(glob(os.path.join(path, '*.pth')))[0]
            assert os.path.isfile(file_path), '=> No checkpoint found at {}'.format(path)
            checkpoint = torch.load(file_path, map_location='cpu')
            model.load_state_dict(checkpoint['model'])
        return model
    def __get_transform(self):
        '''
        get image augmentation
        '''
        transform=[]
        transform.append(T.Resize((224,224)))
        transform.append(T.ToTensor())
        transform.append(T.Normalize(mean=[0.485, 0.456, 0.406] ,std=[0.229, 0.224, 0.225]))
        transform = T.Compose(transform)
        return transform
    def __crop_face(self,img):
        '''
        crop out face area
        '''
        boxes, _ = self.__mtcnn.detect(img)
        if boxes is None:
            return None
        face_img = img.crop(boxes[0].tolist())
        return face_img
    def __predict(self,img):
        '''
        run inference throught model
        '''
        img = self.__crop_face(img)
        if img is None:
            return 1.0,1.0
        spooftype = None
        self.__model = self.__model.to(self.__device)
        img = self.__transform(img).to(self.__device).unsqueeze(0)
        self.__model.eval()
        score = torch.sigmoid(self.__model(img))
        spooftype = torch.gt(score,0.4).long().detach()
        return score.item(),spooftype
    # def base64toPIL(self,img):
    #     pil_img = Image.open(BytesIO(base64.b64decode(img)))
    #     pil_img.save('/home/tintinw/eKYC/backend/controller/APP/PAD/img.jpg')
    #     return pil_img

    def get_score(self,img):
        '''
        get prediction score
        '''
        # img = self.base64toPIL(img)
    
        output = self.__predict(img)
        score,spoof_type = output
        score = 1-score
        return score,spoof_type

if __name__ == "__main__":
    from PIL import Image
    detector = eKYC_PAD()
    img = Image.open('/home/tintinw/eKYC/backend/controller/test_data/0005_selfie_img.jpg')
    score,spooftype = detector.get_score(img)
    print(score)