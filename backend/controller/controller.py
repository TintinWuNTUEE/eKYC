import json
from channels.generic.websocket import AsyncWebsocketConsumer
from .APP.sam import SAMSegmentor
from .APP.IDFD.IDFD import eKYC_IDFD
from .APP.FV.eKYC_FV import eKYC_FV
from .APP.PAD.PAD import eKYC_PAD 
from .APP.FIDD.holodetect import HoloDetector as FIDD
from .APP.OCR.OCR_client import Ocrclient as OCR
from channels.layers import get_channel_layer
import asyncio
import time
import queue
from multiprocessing import Process, Manager
import threading
import numpy as np
from tqdm.auto import tqdm
from .utils import ctime, base64toPIL, UtoD, PILtobase64, b64toCv2
from .configs import get_cfg
from PIL import Image
import cv2
import concurrent.futures
import janus
import logging
import torch

from facenet_pytorch import MTCNN, InceptionResnetV1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(device=device)
resnet = InceptionResnetV1(pretrained='vggface2').to(device).eval()
FV = eKYC_FV(device, mtcnn, resnet)
PAD = eKYC_PAD(mtcnn)

cfg = get_cfg()
cfg.freeze()
segmentor = SAMSegmentor(np.array(cfg.SEGMENT.INPUTBOX),cfg.device,cfg.SEGMENT.CHKPT)
print('Model loaded.')

class Controller():
    def __init__(self, *args, **kwargs):
        # super().__init__(self)
        self.logger = logging.getLogger("ekyc_logger")
        
        self.Stage_dict = {
            '1' : self.Stage_1,
            '2' : self.Stage_2,
            '3' : self.Stage_3,
            '4' : self.Stage_4,
            '5' : self.Stage_5,
            '6' : self.Stage_6,
            '7' : self.Stage_7,
        }
        self.test_dict = {
            'OCR' : [self.Stage_4, self.Stage_5], 
            'FV' : [self.Stage_2, self.Stage_6],
            'PAD' : [self.Stage_6], 
            'FIDD' : [self.Stage_3],
            'IDFD' : [self.Stage_2],
        }
        self.ID_img = {
            'f_1' : None,
            'f_2' : None,
            'f_3' : None,
            'b'   : None,
        }
        self.result = {
            'Session_ID'    : None,
            'Next_Stage'    : None,
            'Score'         : None,
            'Img_ID_f'      : None,
            'Img_ID_b'      : None,
            'Img_Selfie'    : None,
            'Message'       : None,
        }
        self.queue = queue.Queue(maxsize=200)
        self.iteration = 0
        self.stage3_flag = True
        self.FIDD_front_img = []
        self.filter_points = 0
        self.FIDD_score = 0
        self.IDFD_score = 0
        self.task1 = None
    
    async def processing(self, clientID, rec_json):
        self.clientID = clientID
        self.json_dict = json.loads(rec_json)
        if self.json_dict.get('Img', None):
            if self.json_dict['Stage'] == '3':
                self.decode_img = b64toCv2(self.json_dict['Img'].split(',')[1])
            else :
                self.decode_img = base64toPIL(self.json_dict['Img'].split(',')[1])
        await self.Stage_dict[self.json_dict['Stage']]()

    async def Stage_1(self):
        # self.logger.info('test in stage 1.')
        self.pInfo(f'Current Stage: {self.json_dict["Stage"]}')
        self.Session_ID = self.json_dict['Session_ID']
        self.__ID_info = {
            'Name': self.json_dict['Name'],
            'Date': UtoD(self.json_dict['Date']),
            'ID': self.json_dict['ID'], 
        }
        self.pInfo(f'ID Info: {self.__ID_info}')
        await self.__send_json(self.response_json(Next_Stage=2))
    
    async def Stage_2(self):
        self.pInfo(f'Current Stage: {self.json_dict["Stage"]}, imgsize:{self.decode_img.size}')
        self.ID_img = { 'f_1' : self.decode_img, }
        task1 = asyncio.create_task(self.__cal_IDFD(self.ID_img['f_1']))      
        await self.__send_json(self.response_json(Next_Stage=3))
    
    async def Stage_3(self):
        if self.stage3_flag:
            self.pInfo(f'Current Stage: {self.json_dict["Stage"]}, imgsize:{self.decode_img.shape}')
            self.tqdm_progress = tqdm(desc='recv img', total=200)
            self.stage3_flag = False
            # task = asyncio.create_task(self.__cal_FIDD())
            '''
            Threading code
            '''
            self.thread_loop = asyncio.new_event_loop()
            t = threading.Thread(target=self.__cal_FIDD_loop, args=(self.thread_loop, self.queue, ))
            t.daemon = True
            t.start()

        # asyncio.run_coroutine_threadsafe(self.queue.put(self.decode_img), self.thread_loop)
        # await self.queue.async_q.put(self.decode_img)
        # cv2.imwrite(f'backend/controller/test_data/FIDD_q_in/{self.clientID}_{self.iteration}.jpg', self.decode_img)
        # self.iteration += 1
        # self.decode_img.save(f'backend/controller/test_data/{self.clientID}_ID_img_f_1.jpg')
        self.queue.put(self.decode_img)
        self.tqdm_progress.update(1)
        
        if self.json_dict['Id_flag']:
            self.FIDD_front_img.append(self.decode_img)

        if self.json_dict['Last']:
            await self.__send_json(self.response_json(Next_Stage=4))
            self.tqdm_progress.close()
    
    async def Stage_4(self):
        # self.pInfo(f'Current Stage: {self.json_dict["Stage"]}, imgsize:{self.decode_img.size}')
        # self.decode_img.save('raw.jpg')
        segmented_img = await self.__segment(self.decode_img)
        # segmented_img.save('test.jpg')
        self.pInfo(f'Current Stage: {self.json_dict["Stage"]}, receive imgsize:{self.decode_img.size}, Send OCR imgsize:{segmented_img.size}')
        self.ID_img['f_3'] = self.decode_img
        # self.ID_img['f_3'].save(f'./backend/controller/test_data/{self.clientID}_ID_img.jpg')
        self.OCR_score = 0
        self.OCR_Result_f = None
        ###
        await self.__cal_OCR(PILtobase64(segmented_img), self.__ID_info, 1)
        ###
        await self.__send_json(self.response_json(Next_Stage=5))
    
    async def Stage_5(self):
        self.pInfo(f'Current Stage: {self.json_dict["Stage"]}, imgsize:{self.decode_img.size}')
        segmented_img = await self.__segment(self.decode_img)
        self.pInfo(f'Current Stage: {self.json_dict["Stage"]}, Send OCR imgsize:{segmented_img.size}')
        self.ID_img['b'] = self.decode_img
        self.OCR_score = 0
        self.OCR_Result_b = None
        ###
        await self.__cal_OCR(PILtobase64(segmented_img), self.__ID_info, 2)
        ###
        await self.__send_json(self.response_json(Next_Stage=6))
    
    async def Stage_6(self):
        self.pInfo(f'Current Stage: {self.json_dict["Stage"]}, imgsize:{self.decode_img.size}')
        self.selfie_img = self.decode_img
        # self.ID_img_f_1.save(f'backend/controller/test_data/{self.clientID}_ID_img_f_1.jpg')
        # self.selfie_img.save(f'./backend/controller/test_data/{self.clientID}_selfie_img.jpg')
        # self.selfie_img = Image.open(f'./backend/controller/test_data/{self.clientID}_selfie_img.jpg')
        # self.PAD_score,_ = await self.__cal_PAD(self.selfie_img)
        (self.PAD_score,_),self.FV_score = await asyncio.gather(self.__cal_PAD(self.selfie_img), self.__cal_FV(self.selfie_img, self.ID_img['f_1']))
        self.pInfo(f'PAD_Score: {self.PAD_score}')
        self.pInfo(f'FV_Score: {self.FV_score}')
        await self.Stage_7()

    async def Stage_7(self):
        score,flag = self.__cal_FSE()
        result = {
            'Session_ID'    : self.json_dict['Session_ID'],
            'Next_Stage'    : 7,
            'Score'         : score,
            'img_Id_f'      : PILtobase64(self.ID_img['f_1']),
            'img_Id_b'      : PILtobase64(self.ID_img['b']),
            'img_Selfie'    : PILtobase64(self.selfie_img),
            'Message'       : self.OCR_Result_f,
        }
        self.pInfo(f'Final Score: {score}, OCR: {self.OCR_Result_f}')
        await self.__send_json(result)

    def response_json(self, Next_Stage, Message = None):
        re_json = {
            'Session_ID'    : self.json_dict['Session_ID'],
            'Current_Stage' : self.json_dict['Stage'],
            'Next_Stage'    : Next_Stage,
        }
        if Message is not None:
            re_json['Message'] = Message
        return re_json
    
    def pInfo(self, text):
        self.logger.info(f'[{self.clientID}] {text}')
        # print(f'[{ctime()}][Info][{self.clientID}]', text)

    async def __cal_IDFD(self,img):
        await asyncio.sleep(0.1)
        loop = asyncio.get_event_loop()
        detector = eKYC_IDFD()
        try:
            with concurrent.futures.ThreadPoolExecutor() as pool:
                self.IDFD_score,_ = await loop.run_in_executor(pool, detector.get_score, img)
                self.pInfo(f'Stage 2 finished!, IDFD_score: {self.IDFD_score}')

        except Exception as e:
            print(e)
        # return score,spooftype
    
    async def __cal_PAD(self, img):
        await asyncio.sleep(0.1)
        try:
            PAD_score,spooftype = PAD.get_score(img)
        except Exception as e:
            print(e)
        return PAD_score,spooftype
    
    async def __cal_FV(self, img1, img2):
        await asyncio.sleep(0.1)
        # detector = eKYC_FV()  
        try:
            FV_score =  FV.get_score(img1, img2) 
        except Exception as e:
            print(e)
        return FV_score
    # async def __Que(self, img)
    #     self.queue.put(img)

    # def side_thread(self, loop):
    #     asyncio.set_event_loop(loop)
    #     loop.run_

    def __cal_FIDD_loop(self, loop, queue):
        # loop_m = asyncio.get_event_loop()
        asyncio.set_event_loop(loop)
        
        # loop.run_forever()
        FIDD_detector = FIDD()
        async def cal_FIDD():
            try:
                # with concurrent.futures.ThreadPoolExecutor() as pool:
                filter_points = 0
                flag = False
                for i in tqdm(range(200), desc='FIDD'):
                    frame = self.queue.get()
                    # cv2.imwrite(f'backend/controller/test_data/FIDD_q_out/{self.clientID}_{i}.jpg', frame)
                    _, _, points = await FIDD_detector.detect_holos(frame)
                    # future = asyncio.run_coroutine_threadsafe(FIDD_detector.detect_holos(frame), loop)
                    # _, _, points = future.result()
                    if points > filter_points:
                        filter_points = points
                    flag = (5000>filter_points>750)
                self.FIDD_score = filter_points
                self.FIDD_Flag = flag
                self.pInfo(f'FIDD score: {self.FIDD_score}, {self.FIDD_Flag}')
            except Exception as e:
                print(e)
        # loop.stop()
        loop.run_until_complete(cal_FIDD())

    async def __cal_FIDD(self):
        FIDD_detector = FIDD()
        try:
            for i in tqdm(range(200), desc='FIDD'):
                frame = await self.queue.get()
                
                _, _, points = await FIDD_detector.detect_holos(frame)
                if points > self.filter_points:
                    self.filter_points = points
                self.FIDD_Flag = (5000>self.filter_points>750)
            self.FIDD_score = self.filter_points
            self.pInfo(f'FIDD score: {self.FIDD_score}, {self.FIDD_Flag}')
        except Exception as e:
            print(e)
        return 
    
    async def __cal_OCR(self, img, info, flag):
        detector = OCR()
        try:
            if flag == 1:
                self.OCR_Result_f = await detector.client(img, info, flag)
            elif flag == 2:
                self.OCR_Result_b = await detector.client(img, info, flag)
                self.OCR_Result_f.update(self.OCR_Result_b)
            else:
                await detector.client(img, info, flag)
        except Exception as e:
            print(e)

    async def __segment(self,img):
        image = cv2.resize(segmentor.segment(np.array(img)),(832,512),interpolation=cv2.INTER_LANCZOS4)
        image = Image.fromarray(image)
        return image
    
    def pass_threshold(self,thresholds):
        '''
        Check if every score is above threshold
        '''
        return (self.PAD_score >= thresholds.THRES.PAD) \
            and (self.FV_score >= thresholds.THRES.FV)\
            and (self.FIDD_score >= thresholds.THRES.FIDD)\
            and (self.OCR_score >= thresholds.THRES.OCR)\
            and (self.IDFD_score >= thresholds.THRES.IDFD)
    
    def __cal_FSE(self):
        '''
        Weighted sum of every score
        '''
        cfg = get_cfg()
        cfg.freeze()
        flag = None
        if not self.pass_threshold(cfg):
            flag = False
            return 0.0, flag
        score = cfg.WEIGHTS.PAD * self.PAD_score+\
                cfg.WEIGHTS.FV * self.FV_score+\
                cfg.WEIGHTS.FIDD * int(self.FIDD_Flag)+\
                cfg.WEIGHTS.OCR * self.OCR_score+\
                cfg.WEIGHTS.IDFD * self.IDFD_score
        score = float("%.1f"%score)
        return score,flag

    async def __send_json(self, response):
        channel_layer = get_channel_layer()
        await channel_layer.group_send(
            self.clientID,
            {
                "type": "response.message",
                "json": response,
            },
        )

    # async def response_message(self, event):
    #     await self.send(text_data=json.dumps(event["json"]))
    