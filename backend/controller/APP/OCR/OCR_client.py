import asyncio, json, base64
import websockets
from channels.generic.websocket import AsyncWebsocketConsumer
import datetime
class Ocrclient(AsyncWebsocketConsumer):
    def __init__(self):
        super().__init__()

    async def client(self, imgBase64Str, info, flag):
        
        img_data = 'data:image/jpeg;base64,'+ imgBase64Str
        request = await self.get_dict(flag, info)
        request['img'] = img_data
        # img_data = imgBase64Str

        async with websockets.connect("ws://34.81.210.82:6666/") as websocket:
            await websocket.send(json.dumps(request))
            response = await websocket.recv()
            response = json.loads(response)
            # print(f"Received OG response: {response}")
            if flag == 1:
                if response['code'] == '99':
                    rt_data = {
                    'res' : response['msg'],
                    'Name': 'ERROR',
                    'Date': 'ERROR',
                    'ID': 'ERROR'
                }
                else :
                    data = [int(i) for i in response['fields'][1]['birthday'].split(',')]
                    # data[0] += 1911
                    # try:
                    #     date =  datetime.datetime(*data[0:3])
                    # except:
                    #     date = 'NOT DATETIME'

                    rt_data = {
                        'res' : response['msg'],
                        'Name': response['fields'][0]['name'],
                        'Date': f'{data[0]}年{data[1]}月{data[2]}日',
                        'ID': response['fields'][2]['idnum'],
                    }
            elif flag == 2:
                if response['code'] == '99':
                    rt_data = {
                        'res' : response['msg'],
                        'Dad': 'ERROR',
                        'Mom': 'ERROR',
                        'Barcode': 'ERROR',
                        'Mate': 'ERROR'
                    }
                else:
                    rt_data = {
                        'res' : response['msg'],
                        'Dad': response['fields'][0]['father'],
                        'Mom': response['fields'][1]['mother'],
                        'Barcode': response['fields'][3]['idnum'],
                        'Mate': response['fields'][2]['spouse']
                    }
            print(f"Received response: {rt_data}")
            return rt_data
            
    async def get_dict(self, flag, info):
        date = str(int(info['Date'].split('-')[0])-1911)+''.join(info['Date'].split('-')[1:])
        request = {
            "class":"PID",
            "page": str(flag),
            "IDNo": info['ID'],
        }
        if flag == 1:
            request["Name"] = info['Name']
            request["BirthDay"] = date
        return request

if __name__ == "__main__":
    ocrclient = Ocrclient()
    asyncio.run(ocrclient.client())

