from django.shortcuts import render,HttpResponse,redirect
from PIL import Image
import io
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import base64
from pathlib import Path
import os
from django.contrib.auth.models import User
from django.contrib.auth import authenticate
from django.contrib.auth import login
BASE_DIR = Path(__file__).resolve().parent.parent

def home(request):
    return render(request, 'index.html')

def login_user(request):
    if request.method == 'POST':
        username=request.POST.get('username')
        password=request.POST.get('pswrd')
        user=authenticate(request,username=username,password=password)
        if user is not None:
            login(request,user)
            return redirect('select')
        else:
            return HttpResponse ("Useername or password is incorrect!!!!!")
    return render(request, 'login.html')

def preview(request):
    return render(request, 'select.html')
def signup(request):
    if request.method == 'POST':
        uname_first = request.POST.get('firstname')
        uname_last = request.POST.get('lastname')
        email = request.POST.get('email')
        pass0 = request.POST.get('password1')
        pass1 = request.POST.get('password2')
        print(uname_first,uname_last,email,pass1)
        if pass0 == pass1:
            #return HttpResponse("Your password and confrom password are not Same!!")
            # Define the username based on your desired logic
            username = f"{uname_first.lower()}.{uname_last.lower()}"
            my_user = User.objects.create_user(username=username, email=email, password=pass0)
            my_user.first_name = uname_first
            my_user.last_name = uname_last
            my_user.save()
            # Redirect the user to some other page
            return redirect('index') 
        
    return render(request, 'signup.html')

def result(request):
    error = 0
    result = None
    img_byte_arr = None
    
    if request.method == 'POST':
        try:
            # Load your custom brain tumor classification model
            model = tf.keras.models.load_model(str(BASE_DIR / 'mri/models.h5'), compile=False)
            
            img = request.FILES.get('image')
            img = Image.open(img)
            realimg = img
            img_byte_arr = io.BytesIO()
            # Define quality of saved array
            realimg.save(img_byte_arr, format='JPEG', subsampling=0, quality=100)
            # Converts image array to bytesarray
            img_byte_arr = base64.b64encode(img_byte_arr.getvalue()).decode('UTF-8')
            
            img = np.asarray(img)
            cpyimg = img
            dim = (224, 224)
            img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
            X = image.img_to_array(img)
            X = np.expand_dims(X, axis=0)
            if X.shape[3] == 1:
                img = cpyimg
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
                X = image.img_to_array(img)
                X = np.expand_dims(X, axis=0)
            
            # Preprocess the image using the same preprocessing as during training
            X = preprocess_input(X)
            
            # Make predictions using the loaded model
            result = model.predict(X)
            print("Raw prediction array:", result)
            
            # Get the index of the predicted class
            predicted_class_idx = np.argmax(result)
            print("Argmax of prediction:", predicted_class_idx)
            
            category = ['Glioma Tumor', 'Meningioma Tumor', 'No Tumor', 'Pituitary Tumor']
            result = category[predicted_class_idx]
            error = 1
        except Exception as e:
            print("Exception:", e)
            error = 2

    return render(request, 'result.html', {'result': result, 'img': img_byte_arr, 'error': error})