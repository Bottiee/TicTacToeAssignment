from django.db import models
from django.contrib.auth.models import User

class Score(models.Model):
    player_name = models.CharField(max_length=100)
    score = models.IntegerField()
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='score')  # <--- important here
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.player_name}: {self.score}"
