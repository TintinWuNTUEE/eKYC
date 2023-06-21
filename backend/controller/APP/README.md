# Presentation Attack Detection(PAD)
    活體辨識
    辨識照片內人物是否為翻拍：
        1. 紙張翻拍
        2. 螢幕翻拍
        3. 3D橡膠模型翻拍
    
## How to use
```python 
class eKYC_PAD:
detector = eKYC_PAD()
score,spooftype = detector.get_score(pil_img)
```
## Public Functions
```python 
    def get_score(self,img):
        '''
        取得PAD過sigmoid後的成績
        於0~1之間，代表做出判斷的信心值
        '''
        input: Img(PIL)
        return:
            score:[0,1],float
            spooftype: 0 or 1, int
```
## Private Functions
```python 
    def __get_device(self):
        '''
        取得可使用的運算資源：CPU/GPU/MPS
        '''
        return: torch.device
    self.device = self.__get_device()
    def __load_model(self,model,path):
        '''
        讀取訓練後的pth模型參數檔
        '''
        input: Model(torchvision.models), Path('backend/.../efficient.pth')
        return: torchvision.models

    path 'backend/.../efficient.pth'
    model = self.__load_model(model,path)
    def __get_transform(self):
        '''
        取得資料預處理的公式
        '''
        return: torch.Compose(transforms)
    transforms = self.__get_transforms()
    def __crop_face(self,img):
        '''
        從圖片中切出正臉
        '''
        input: Img(PIL)
        face_img = img.crop(boxes[0].tolist())
    
        return: Img(PIL) or None(if no face in img)
    face_img = self.__crop_face(img)
    def __predict(self,img):
        '''
        將照片放進模型中進行運算
        '''
        input: Img(PIL)
        return:
            score:[0,1],float
            spooftype:0 or 1, int
    score,spooftype = self.__predict(img)
```
# Fake ID Detection(FIDD)
    輸入一段影片，
    計算出照片中的身分證，哪些部分包含防偽標籤，
    以及回傳總防偽標籤像素點數
    目前設置的閥值為超越750且小於5000
    （若高於上界可能為過曝、低於下界則為防偽標籤點數過少）
## How to use
```python 
class eKYC_FIDD:
detctor = eKYC_FIDD()
queue = Queue.queue()
frame = queue.get()
if frame is not None:
    mask,masked_holo,filter_point = detector.detect_holo(frame)
```
## Public Functions
``` python
    def detect_holo(self,img):
        '''
        偵測防偽標籤位置，回傳mask、mask後的疊圖、總共算出多少點
        '''
        input: Img(cv2)
        return: holo_mask, img_holo, filtered_points
    def detect_rect(img):
        '''
        取得身分證長方形
        '''
        input: Img(cv2,grayscaled)
        return: rect(numpy array)
    def calc_filtered_points():
        '''
        取得有明顯光線變化的點進行過濾後回傳
        '''
        return filtered_points(numpy array)
    def calc_holo_points():
        '''
        轉換為可使用的pixel點
        '''
        return: len(filtered_points)
```
# Segment Anything Model(SAM)
    使用Meta開源之Segment Anything Model為模型，
    將使用者拍攝之身分證進行去背景、取前景、翻轉至水平的以上三個動作
## How to use
``` python
class SAM:
    
segmentor = SAM()
rec_img = Image.open('sample.jpg')
img = segmentor.segment(rec_img)
``` 
## Public Functions
``` python
    def segment(self,img):
        '''
        取得去背景後的身分證
        '''
        input: Img(PIL)
        return: Img(PIL)
```
## Private Functions
``` python
    def __crop_image(self,img):
        '''
        將身分證從背景取出，並進行幾何校正至水平
        '''
        input: Img(PIL)
        return: Img(PIL)
    def __seg2bbox(self,mask):
        '''
        將切割後的mask轉為bounding box
        '''
        input: numpy array(segmentation mask)
        return: numpy array
```
