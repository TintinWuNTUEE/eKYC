import torch
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import math



class eKYC_FV:
    def __init__(self, device, mtcnn, resnet) :
        '''
        init
        '''
        self.__device = device
        self.__mtcnn = mtcnn
        self.__resnet = resnet
        # self.__device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.__mtcnn = MTCNN(device=self.__device)
        # self.__resnet = InceptionResnetV1(pretrained='vggface2').to(self.__device).eval()
         
    def __get_face_emb(self, img):
        '''
        Get face embanding
        '''
        with torch.no_grad():
            face = self.__mtcnn(img)
            if face is None:
                return None
            face = face.to(self.__device)
            face_emb = self.__resnet(face.unsqueeze(0))
        return face_emb.cpu().numpy()
    
    def __distance(self, embeddings1, embeddings2, distance_metric=0):
        if distance_metric==0:
            # Euclidian distance
            diff = np.subtract(embeddings1, embeddings2)
            dist = np.sum(np.square(diff),1)
        elif distance_metric==1:
            # Distance based on cosine similarity
            dot = np.sum(np.multiply(embeddings1, embeddings2), axis=1)
            norm = np.linalg.norm(embeddings1, axis=1) * np.linalg.norm(embeddings2, axis=1)
            similarity = dot / norm
            dist = np.arccos(similarity) / math.pi
        else:
            raise 'Undefined distance metric %d' % distance_metric

        return dist[0]
    
    def __findCosineDistance(self, source_representation, test_representation):
        a = np.matmul(np.transpose(source_representation), test_representation)
        b = np.sum(np.multiply(source_representation, source_representation))
        c = np.sum(np.multiply(test_representation, test_representation))
        return 1 - (a / (np.sqrt(b) * np.sqrt(c)))
        
    def get_score(self, img_IDcard, img_Selfie):
        
        IDcard_face_emb = self.__get_face_emb(img_IDcard)
        Selfie_face_emb = self.__get_face_emb(img_Selfie)
        if IDcard_face_emb is None or Selfie_face_emb is None:
            score = 0
            return score
        distance = self.__distance(IDcard_face_emb, Selfie_face_emb, distance_metric=1)
        # distance = self.__findCosineDistance(IDcard_face_emb[0].tolist(), Selfie_face_emb[0].tolist())
        score = 1-distance
        return score

    def __sigmoid(self, x):
        return 1 / (1 + math.exp(-x))
        
# FV = eKYC_FV()

if __name__ == '__main__':
    FV = eKYC_FV()
    img1 = Image.open('/home/tintinw/eKYC/backend/controller/APP/FV/test_data/Ken_IDcard.jpg')
    img2 = Image.open('/home/tintinw/eKYC/backend/controller/APP/FV/test_data/Ken_Selfie.jpg')
    print(FV.get_score(img1,img2))