from django.urls import re_path
from controller import consumers

websocket_urlpattern=[
    re_path('wss/', consumers.eKYC_Consumer.as_asgi())
]