from django.contrib import admin
from django.urls import path
from analyzer import views

urlpatterns = [
  path('', views.analyze, name='analyze'),  # This should match the root URL
]
