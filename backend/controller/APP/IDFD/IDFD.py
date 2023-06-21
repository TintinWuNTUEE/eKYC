import os
from glob import glob
import torch
from torchvision import transforms as T
from .model.model import get_model
from PIL import Image
class eKYC_IDFD:
    def __init__(self) :
        '''
        init
        '''
        self.__device=self.__get_device()
        self.__models = []
        self.__paths = ['/home/tintinw/eKYC/backend/controller/APP/IDFD/weights_efficient']
        model = get_model()
        self.__models.append(self.__load_model(model,self.__paths[0]))
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
    def __predict(self,img):
        '''
        run inference throught model
        '''
        # img = self._read_img(img)
        # img = self.__crop_face(img)
        if img is None:
            return 1,1
        scores = []
        spooftype = None
        for i,model in enumerate(self.__models):
            model = model.to(self.__device)
            img = self.__get_transform()(img).to(self.__device).unsqueeze(0)
            model.eval()
            score = torch.sigmoid(model(img))
            spooftype = torch.gt(score,0.4).long().detach()
            score = score.item()
            scores.append(score.item())
        del img
        torch.cuda.empty_cache()
        return scores[0],spooftype
    def get_score(self,img):
        '''
        get prediction score
        '''
        output = self.__predict(img)
        score,spoof_type = output
        score = 1-score
        return score,spoof_type