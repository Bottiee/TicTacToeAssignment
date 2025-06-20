import requests
import json
from kivy.uix.screenmanager import Screen
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.scrollview import ScrollView
from kivy.uix.gridlayout import GridLayout
from kivy.uix.button import Button
from kivy.utils import get_color_from_hex
from kivy.clock import mainthread
import threading
from kivy.app import App

from tictactoe_django_integration.leaderboard.utils.history_storage import HistoryStorage


class LeaderboardScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = 'leaderboard'

        # Main layout
        layout = BoxLayout(orientation='vertical', padding=20, spacing=10)

        # Title
        title = Label(
            text='Leaderboard',
            font_size='32sp',
            size_hint=(1, 0.1),
            color=get_color_from_hex('#FFD700')
        )
        layout.add_widget(title)

        # Scrollable area
        scroll = ScrollView(size_hint=(1, 0.7))
        self.grid = GridLayout(
            cols=2,
            spacing=10,
            size_hint_y=None,
            padding=[10, 10, 10, 10]
        )
        self.grid.bind(minimum_height=self.grid.setter('height'))
        scroll.add_widget(self.grid)
        layout.add_widget(scroll)

        # Sync History Button
        sync_btn = Button(
            text='Sync Game History to Django',
            size_hint=(1, 0.1),
            background_color=get_color_from_hex('#4CAF50'),
            color=(1, 1, 1, 1)
        )
        sync_btn.bind(on_release=self.trigger_sync_history)
        layout.add_widget(sync_btn)

        # Back button
        back_btn = Button(
            text='Back to Options',
            size_hint=(1, 0.1),
            background_color=get_color_from_hex('#2196F3'),
            color=(1, 1, 1, 1)
        )
        back_btn.bind(on_release=self.go_back)
        layout.add_widget(back_btn)

        self.add_widget(layout)
        self.load_users()

    def load_users(self):
        url = "http://127.0.0.1:8000/api/users/"
        app = App.get_running_app()
        headers = {}

        if hasattr(app, 'user_token') and app.user_token:
            headers['Authorization'] = f'Token {app.user_token}'

        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            users_data = response.json()

            self.grid.clear_widgets()
            self.grid.add_widget(self.create_label("Username", bold=True))
            self.grid.add_widget(self.create_label("Total Games Played", bold=True))

            if not users_data:
                self.grid.add_widget(self.create_label("No users found.", italic=True))
                self.grid.add_widget(self.create_label(""))
            else:
                for user_info in users_data:
                    username = user_info.get('username', 'N/A')
                    total_games_played = user_info.get('total_games_played', 0)
                    self.grid.add_widget(self.create_label(username))
                    self.grid.add_widget(self.create_label(str(total_games_played)))

        except requests.exceptions.RequestException as e:
            self.grid.clear_widgets()
            self.grid.add_widget(self.create_label("Error loading leaderboard: Ensure Django server is running.", bold=True))
            self.grid.add_widget(self.create_label(""))
            print(f"Leaderboard load error: {e}")
        except json.JSONDecodeError:
            self.grid.clear_widgets()
            self.grid.add_widget(self.create_label("Invalid JSON response from Django server.", bold=True))
            self.grid.add_widget(self.create_label(""))

    def trigger_sync_history(self, *args):
        self.grid.clear_widgets()
        self.grid.add_widget(self.create_label("Syncing history...", bold=True))
        self.grid.add_widget(self.create_label(""))
        threading.Thread(target=self._perform_sync_request).start()

    def _perform_sync_request(self):
        url = "http://127.0.0.1:8000/api/sync-history/"
        app = App.get_running_app()
        headers = {}
        if hasattr(app, 'user_token') and app.user_token:
            headers = {'Authorization': f'Token {app.user_token}'}

        username = getattr(app, 'current_username', None)
        if not username:
            self.update_sync_status_ui("Sync Failed: No logged-in user found.", is_error=True)
            return

        local_history_instance = None
        try:
            kivy_history_db_path = 'Game_history/game_data/history.db'
            local_history_instance = HistoryStorage(kivy_history_db_path)

            data_to_send = {
                'username': username,
                'total_games': local_history_instance.total_games,
                'player1_wins': local_history_instance.player1_wins,
                'player2_wins': local_history_instance.player2_wins,
                'cpu_wins': local_history_instance.cpu_wins,
                'ties': local_history_instance.ties,
            }

            response = requests.post(url, json=data_to_send, headers=headers)

            print(f"\n--- Django Sync Response ---")
            print(f"Status Code: {response.status_code}")
            print(f"Response Body: {response.text}")
            print(f"----------------------------\n")

            response.raise_for_status()
            result = response.json()

            self.update_sync_status_ui(
                result.get('success', result.get('error', 'Unknown response from server.')),
                is_error='error' in result
            )
            self.load_users_on_mainthread()

        except requests.exceptions.ConnectionError:
            self.update_sync_status_ui("Connection Error: Could not connect to Django server.", is_error=True)
        except requests.exceptions.Timeout:
            self.update_sync_status_ui("Request timed out.", is_error=True)
        except requests.exceptions.HTTPError as e:
            error_details = ""
            try:
                error_json = e.response.json()
                error_details = error_json.get('error', error_json.get('detail', ''))
            except json.JSONDecodeError:
                error_details = e.response.text
            self.update_sync_status_ui(f"Sync Failed ({e.response.status_code}): {error_details}", is_error=True)
        except json.JSONDecodeError:
            self.update_sync_status_ui("Invalid JSON response from server.", is_error=True)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.update_sync_status_ui(f"Unexpected error: {e}", is_error=True)
        finally:
            if local_history_instance:
                local_history_instance.close()

    @mainthread
    def update_sync_status_ui(self, message, is_error=False):
        self.grid.clear_widgets()
        status_label = self.create_label(message, bold=True)
        status_label.color = get_color_from_hex('#FF0000') if is_error else get_color_from_hex('#00FF00')
        self.grid.add_widget(status_label)
        self.grid.add_widget(self.create_label(""))

    @mainthread
    def load_users_on_mainthread(self):
        self.load_users()

    def go_back(self, *args):
        self.manager.current = 'options'

    def create_label(self, text, bold=False, italic=False):
        label = Label(
            text=text,
            bold=bold,
            italic=italic,
            color=get_color_from_hex('#FFD700'),
            size_hint_y=None,
            height=40,
            padding=(10, 10),
            halign='center',
            valign='middle'
        )
        label.bind(size=label.setter('text_size'))
        return label
