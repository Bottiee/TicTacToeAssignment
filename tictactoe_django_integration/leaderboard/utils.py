from django.contrib.auth.models import User
from leaderboard.models import Score
from .utils.history_storage import HistoryStorage

def sync_local_history_to_db(username: str):
    # Open the local SQLite history file (adjust path accordingly)
    history = HistoryStorage()

    try:
        # Find the Django user
        user = User.objects.get(username=username)
    except User.DoesNotExist:
        print(f"User '{username}' not found in Django DB.")
        return

    # Get total games played from local history
    total_games = history.total_games

    # Update or create the Score record for this user
    score_obj, created = Score.objects.update_or_create(
        user=user,
        defaults={'total_games_played': total_games}
    )

    if created:
        print(f"Created new Score for {user.username} with {total_games} games played.")
    else:
        print(f"Updated Score for {user.username} to {total_games} games played.")

    # Always good to close DB connections
    history.close()
