# C:\Users\botyl\PycharmProjects\KryziukaiNuliukaiUzduotis\tictactoe_django_integration\leaderboard\urls.py

from django.urls import path
from . import views
from django.contrib import admin

urlpatterns = [
    path('', views.home_view, name='home'),
    path('admin/', admin.site.urls),
    path('leaderboard/', views.leaderboard_view, name='leaderboard'),
    path('register/', views.register_user, name='register_user_api'),
    path('users/', views.user_list, name='user_list_api'),
    path('sync-history/', views.sync_ttt_history_api, name='sync_history_api'),
    path('users/', views.user_template_view, name='user_template_view'),
]