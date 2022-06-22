from xml.etree.ElementInclude import include
from django.contrib import admin
from django.urls import path ,include
from . import views

app_name="image"
urlpatterns = [
    path('',views.index,name='imghome'),
    path(r'^delete/(?P<pk>[0-9]+)/$',views.delete,name='imgDelete'),
]
