from datetime import datetime,timezone,timedelta
from channels.layers import get_channel_layer
from PIL import Image
from io import BytesIO
import base64
import os
import cv2
import numpy as np
def ctime():
    dt1 = datetime.utcnow().replace(tzinfo=timezone.utc)
    dt2 = dt1.astimezone(timezone(timedelta(hours=8)))
    return dt2.strftime("%Y-%m-%d %H:%M:%S")

def current_groups():
    channel_layer = get_channel_layer()
    print("[current groups]:")
    if channel_layer.groups is None :
        print(f'No Member.')
    else:
        for group in channel_layer.groups:
            print(f'\t{group}')
def b64toCv2(img):
    im = cv2.imdecode(np.frombuffer(base64.b64decode(img),dtype=np.uint8),flags=cv2.IMREAD_COLOR)
    return im
def base64toPIL(img):
    img_buffer = BytesIO()
    pil_img = Image.open(BytesIO(base64.b64decode(img)))
    pil_img.save(img_buffer,format='JPEG')
    img = Image.open(img_buffer)
    return img

def PILtobase64(img):
    img_buffer = BytesIO()
    img.save(img_buffer, format='JPEG',dpi=(300,300))
    base64_str = base64.b64encode(img_buffer.getvalue()).decode()
    return base64_str
    
def base64WebptoPIL(img):
    pil_img = Image.open(BytesIO(base64.b64decode(img))).resize((512,910))
    return pil_img

def UtoD(unix_time):
    return (datetime.utcfromtimestamp(unix_time).strftime('%Y-%m-%d'))

def save_img(img, path):
    if not os.path.exists(path):
        os.mkdir(path)
    # img = Image.open(img)
    img.save(path)