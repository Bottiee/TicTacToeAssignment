# leaderboard/models.py
from django.db import models
from django.contrib.auth.models import User

class Score(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='scores')
    total_games_played = models.IntegerField(default=0)
    player1_wins = models.IntegerField(default=0)
    player2_wins = models.IntegerField(default=0)
    cpu_wins = models.IntegerField(default=0)
    ties = models.IntegerField(default=0)

    def __str__(self):
        return f"{self.user.username}'s Score"

    # You might want to add a method to calculate derived stats if any
    # For example, win_loss_ratio, etc.