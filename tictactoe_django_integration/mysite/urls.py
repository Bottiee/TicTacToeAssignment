from django.urls import path
from leaderboard.views import register_user, user_list
from django.contrib import admin
urlpatterns = [
    path('api/register/', register_user, name='register_user'),
    path('api/users/', user_list, name='user_list'),
    path('admin/', admin.site.urls),
]
