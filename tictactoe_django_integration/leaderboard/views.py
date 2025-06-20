# C:\Users\botyl\PycharmProjects\KryziukaiNuliukaiUzduotis\tictactoe_django_integration\leaderboard\views.py
from django.http import HttpResponse
from django.shortcuts import render
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated, AllowAny  # Import AllowAny
from rest_framework.response import Response
from rest_framework import status
from django.contrib.auth.models import User
from .models import Score

def home_view(request):
    top_scores = Score.objects.select_related('user').order_by('-total_games_played')
    users_data_for_template = []
    for score_obj in top_scores:
        users_data_for_template.append({
            'username': score_obj.user.username,
            'max_games': score_obj.total_games_played
        })
    context = {
        'users': users_data_for_template
    }
    return render(request, 'home.html', context)

# leaderboard_view - (Keep as is, but consider @permission_classes([AllowAny]) if public)
@api_view(['GET'])
# @permission_classes([AllowAny]) # Allows anyone to view the leaderboard. If commented, uses DEFAULT_PERMISSION_CLASSES (IsAuthenticated)
def leaderboard_view(request):
    try:
        scores = Score.objects.select_related('user').order_by('-total_games_played', 'user__username')
        leaderboard_data = []
        for score in scores:
            leaderboard_data.append({
                'username': score.user.username,
                'total_games_played': score.total_games_played,
                'player1_wins': score.player1_wins,
                'player2_wins': score.player2_wins,
                'cpu_wins': score.cpu_wins,
                'ties': score.ties,
            })
        return Response(leaderboard_data, status=status.HTTP_200_OK)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


# register_user - (Keep as is, needs no authentication)
@api_view(['POST'])
@permission_classes([AllowAny])  # Users must be able to register without being logged in
def register_user(request):
    username = request.data.get('username')
    password = request.data.get('password')

    if not username or not password:
        return Response({'error': 'Username and password are required.'}, status=status.HTTP_400_BAD_REQUEST)

    try:
        if User.objects.filter(username=username).exists():
            return Response({'error': 'Username already taken.'}, status=status.HTTP_400_BAD_REQUEST)

        user = User.objects.create_user(username=username, password=password)
        Score.objects.create(user=user)

        return Response({'success': 'User registered successfully!', 'username': user.username},
                        status=status.HTTP_201_CREATED)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

# user_list - (Keep as is, but consider @permission_classes([IsAuthenticated]) for security)
@api_view(['GET'])
# @permission_classes([IsAuthenticated]) # Allow only authenticated users to view the user list
def user_list(request):
    try:
        scores = Score.objects.select_related('user').order_by('user__username')
        user_data = []
        for score in scores:
            user_data.append({
                'id': score.user.id,
                'username': score.user.username,
                'total_games_played': score.total_games_played,
                'player1_wins': score.player1_wins,
                'player2_wins': score.player2_wins,
                'cpu_wins': score.cpu_wins,
                'ties': score.ties,
            })
        return Response(user_data, status=status.HTTP_200_OK)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

# sync_ttt_history_api - AUTH BYPASS ENABLED HERE
@api_view(['POST'])
@permission_classes([AllowAny])  # No auth required
def sync_ttt_history_api(request):
    try:
        # Get username from request data to identify user (since no auth)
        username = request.data.get('username')
        if not username:
            return Response({'error': 'Username is required for syncing without authentication.'},
                            status=status.HTTP_400_BAD_REQUEST)

        try:
            user_to_sync = User.objects.get(username=username)
        except User.DoesNotExist:
            return Response({'error': f"User '{username}' does not exist."},
                            status=status.HTTP_404_NOT_FOUND)

        # Retrieve game stats from the request body
        total_games = request.data.get('total_games', 0)
        player1_wins = request.data.get('player1_wins', 0)
        player2_wins = request.data.get('player2_wins', 0)
        cpu_wins = request.data.get('cpu_wins', 0)
        ties = request.data.get('ties', 0)

        if not all(isinstance(x, int) for x in [total_games, player1_wins, player2_wins, cpu_wins, ties]):
            return Response({'error': 'Invalid data types for game stats. Expected integers.'},
                            status=status.HTTP_400_BAD_REQUEST)

        score_obj, created = Score.objects.update_or_create(
            user=user_to_sync,
            defaults={
                'total_games_played': total_games,
                'player1_wins': player1_wins,
                'player2_wins': player2_wins,
                'cpu_wins': cpu_wins,
                'ties': ties,
            }
        )

        message = (f"Score for {user_to_sync.username} {'created' if created else 'updated'} to: "
                   f"Total: {total_games}, P1 Wins: {player1_wins}, P2 Wins: {player2_wins}, "
                   f"CPU Wins: {cpu_wins}, Ties: {ties}.")
        return Response({'success': message}, status=status.HTTP_200_OK)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

def user_template_view(request):
    return HttpResponse("This is the user template view. Render an HTML template here.")
