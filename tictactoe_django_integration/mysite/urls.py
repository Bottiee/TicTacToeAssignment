from django.contrib import admin
from django.urls import path, include
from leaderboard import views  # needed for home_view and leaderboard_view
from leaderboard.views import register_user, user_list

urlpatterns = [
    path('', views.home_view, name='home'),
    path('leaderboard/', views.leaderboard_view, name='leaderboard'),
    path('api/register/', register_user, name='register_user'),
    path('api/users/', user_list, name='user_list'),
    path('admin/', admin.site.urls),
    path('users/', views.user_template_view, name='user_template_view'),

    # Optionally include app-level urls if you have more views there
    # path('', include('leaderboard.urls')),
]
