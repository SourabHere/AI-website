from multiprocessing import Event
from django.shortcuts import get_object_or_404, redirect, render

from image.models import images

def delete(request,pk):
    item=get_object_or_404(images, img_id=pk)
    if request.method=="POST":
        item.delete()
        return redirect("/")
    myitems=images.objects.all()
    return render(request,'image/index.html',{'myitems':myitems})

def index(request):
    function_match={"Face_Detection":0,"Face%":1,"emotion":2,"gender":3,"Mask":4,"Pose":5}
    if request.method=="POST":
        img=request.FILES.get("image","")
        img2=request.FILES.get("image2","")
        img_name=request.POST.get("image_n","")
        choice_func=request.POST.get("Function","")
        choice=function_match[choice_func]
        print(choice)
        if choice==1:
            image=images(image=img,img_name=img_name,image2=img2,choice=choice)
        else:
            image=images(image=img,img_name=img_name,image2="",choice=choice)
        image.save()
        
        myitems=images.objects.all()
        return render(request,'image/index.html',{"myitems":myitems})
    myitems=images.objects.all()
    print(myitems)
    return render(request,'image/index.html',{'myitems':myitems})