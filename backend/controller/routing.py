from django.urls import path
from .consumers import eKYC_Consumer

ws_urlpatterns = [
    path('ws/', eKYC_Consumer.as_asgi())
]