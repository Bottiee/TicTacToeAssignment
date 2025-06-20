# leaderboard/views.py
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.contrib.auth.models import User
from django.db.models import Max
import json
from .models import Score
from .serializers import UserSerializer  # assuming you have this for your API

def home_view(request):
    users_with_scores = User.objects.annotate(
        max_score=Max('score__score')
    ).order_by('-max_score')
    return render(request, 'leaderboard/home.html', {'users': users_with_scores})


def leaderboard_view(request):
    players = Score.objects.order_by('-score')[:10]
    return render(request, 'leaderboard/leaderboard.html', {'players': players})

@csrf_exempt  # for testing; remember to secure this in production!
def register_user(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            username = data.get('username')
            password = data.get('password')
            email = data.get('email', '')

            if User.objects.filter(username=username).exists():
                return JsonResponse({'error': 'Username already exists'}, status=400)

            user = User.objects.create_user(username=username, password=password, email=email)
            user.save()

            return JsonResponse({'success': 'User created successfully'})
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'error': 'Invalid HTTP method'}, status=405)

@api_view(['GET'])
def user_list(request):
    users = User.objects.all()
    serializer = UserSerializer(users, many=True)
    return Response(serializer.data)

def user_template_view(request):
    users = User.objects.all().order_by('username')
    return render(request, 'leaderboard/users.html', {'users': users})
