# from django.contrib.auth import views
from django.urls import path
from .views import register 
from . import views


urlpatterns = [
    # define path for register
    path('register/',register,name='register'),
    path("login/", views.user_login, name="login"),
    path("logout/", views.user_logout, name="logout"),
    # path for deshboard
    path("dashboard/", views.dashboard, name="dashboard"),
    # path for emotion_detection
    path(
        "emotion/",views.emotion_detection,name="emotion_detection"
    ),
    # path for APi call 
     path(
        "api/detect-emotion/",
        views.detect_emotion,
        name="detect_emotion"
    ),
    path("chat/api/", views.chat_api, name="chat_api"),
    # for chat bot 
    path("chat/", views.chat_page, name="chat_page"),
    # path for user chat history
    path(
    "chat-history/",
    views.chat_history,
    name="chat_history"
),
]