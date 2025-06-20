# leaderboard/management/commands/sync_ttt_history.py
from django.core.management.base import BaseCommand
from django.contrib.auth.models import User
from tictactoe_django_integration.leaderboard.models import Score
from tictactoe_django_integration.leaderboard.utils.history_storage import HistoryStorage

class Command(BaseCommand):
    help = 'Synchronizes total games played from local TTT history.db to Django Score model.'

    def add_arguments(self, parser):
        parser.add_argument(
            '--username',
            type=str,
            help='Specify a username to sync. If omitted, attempts to sync for all existing Django users.',
            nargs='?' # Make it optional
        )
        parser.add_argument(
            '--all',
            action='store_true',
            help='Syncs history for ALL existing Django users.',
        )

    def handle(self, *args, **options):
        username_to_sync = options['username']
        sync_all = options['all']

        if username_to_sync:
            self._sync_single_user(username_to_sync)
        elif sync_all:
            self._sync_all_users()
        else:
            self.stdout.write(self.style.WARNING("Please specify --username <name> or --all to sync history."))


    def _sync_single_user(self, username: str):
        history = None
        try:
            user = User.objects.get(username=username)
            self.stdout.write(self.style.SUCCESS(f"Attempting to sync for user: {username}"))

            history = HistoryStorage() # This will connect to history.db
            total_games = history.total_games

            score_obj, created = Score.objects.update_or_create(
                user=user,
                defaults={'total_games_played': total_games}
            )

            if created:
                self.stdout.write(self.style.SUCCESS(f"Created new Score for {user.username} with {total_games} games played."))
            else:
                self.stdout.write(self.style.SUCCESS(f"Updated Score for {user.username} to {total_games} games played."))

        except User.DoesNotExist:
            self.stderr.write(self.style.ERROR(f"User '{username}' not found in Django DB."))
        except Exception as e:
            self.stderr.write(self.style.ERROR(f"Error syncing history for {username}: {e}"))
        finally:
            if history:
                history.close()

    def _sync_all_users(self):
        self.stdout.write(self.style.SUCCESS("Attempting to sync history for all Django users..."))
        for user in User.objects.all():
            self._sync_single_user(user.username)
        self.stdout.write(self.style.SUCCESS("Sync process completed."))