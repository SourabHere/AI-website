from django.contrib import admin
from image.models import images

from sorab.views import index
from django.contrib import admin
from .models import images

# Register your models here.
admin.site.register(images)