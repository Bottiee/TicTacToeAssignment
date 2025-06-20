# mysite/urls.py

from django.contrib import admin
from django.urls import path, include
from rest_framework.authtoken.views import obtain_auth_token
from leaderboard.views import home_view

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', home_view, name='home'),
    path('api/', include('leaderboard.urls')),
    path('api-token-auth/', obtain_auth_token),
]