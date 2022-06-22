from django.shortcuts import render
from django.http import HttpResponse
from requests import request
from  django.http import HttpResponse

from image.models import images


def index(request):
    return render(request,"index.html")