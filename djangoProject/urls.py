"""djangoProject URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [
    path('', views.to_img_load),
    path('histogram/', views.histogram),
    path('greyHistogram/', views.greyHistogram),
    path('colorHistogram/', views.colorHistogram),
    path('piecewiseLinearProcessing/', views.piecewiseLinearProcessing),
    path('enlarge/', views.enlarge),
    path('move/', views.move),
    path('spin/', views.spin),
    path('horizontalFlip/', views.horizontalFlip),
    path('verticalFlip/', views.verticalFlip),
    path('crossFlip/', views.crossFlip),
    path('affineTransformation/', views.affineTransformation),
    path('enhance/', views.enhance),
    path('robs/', views.robs),
    path('sob/', views.sob),
    path('lap/', views.lap),
    path('log/', views.log),
    path('cny/', views.cny),
    path('MeanFilter/', views.MeanFilter),
    path('MedFilter/', views.MedFilter),
    path('HoughLineChange/', views.HoughLineChange),
    path('erode/', views.erode),
    path('dialate/', views.dialate),
    path('sp_noise/', views.sp_noise),
    path('gasuss_noise/', views.gasuss_noise),
    path('highPassFilter/', views.highPassFilter),
    path('IdealLowPassFiltering/', views.IdealLowPassFiltering),
    path('butterworth_low_filter/', views.butterworth_low_filter),
    path('IdealHighPassFiltering/', views.IdealHighPassFiltering),
    path('butterworth_high_filter/', views.butterworth_high_filter),
    path('sharpen/', views.sharpen),
    path('faceDetect/', views.faceDetect)
]
