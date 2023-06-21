from channels.generic.websocket import AsyncWebsocketConsumer
from channels.exceptions import StopConsumer
import json
from .controller import Controller
import uuid
from channels.layers import get_channel_layer
from .utils import ctime, current_groups
import torch
import gc
import logging

class eKYC_Consumer(AsyncWebsocketConsumer):
    clientcount = 0
    def __init__(self):
        super().__init__(self)
        self.logger = logging.getLogger('ekyc_logger')
        self.Controller = Controller()
        # self.clientcount = 0
        
    async def connect(self):
        await self.accept()
        eKYC_Consumer.clientcount += 1
        self.clientID = str("%04d"% eKYC_Consumer.clientcount)
        await self.channel_layer.group_add(self.clientID, self.channel_name)
        self.logger.info(f'[connect] {self.clientID}')
        # print(f'[{ctime()}][New connect client] {self.clientID}')
        # current_groups()


    async def disconnect(self, close_code):
        await self.channel_layer.group_discard(self.clientID, self.channel_name)
        
        del self.Controller
        torch.cuda.empty_cache()
        gc.collect()
        self.logger.info(f'[Disconnect] {self.clientID}')
        # print(f'[{ctime()}][Disconnect] {self.clientID}')
        # current_groups()
        # raise StopConsumer()

    async def receive(self, text_data):
        await self.Controller.processing(self.clientID, text_data)

    async def response_message(self, event):
        await self.send(text_data=json.dumps(event["json"]))
        
        
    
        
