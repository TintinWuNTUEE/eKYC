from django.urls import path
from .views import Index
from django.views.generic.base import TemplateView

urlpatterns = [
    path('', Index.as_view(), name='index'),
    path('manifest.json', TemplateView.as_view(template_name='manifest.json', content_type='application/json'), name='manifest.json')

]