import requests
from kivy.uix.screenmanager import Screen
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.scrollview import ScrollView
from kivy.uix.gridlayout import GridLayout
from kivy.uix.button import Button
from kivy.utils import get_color_from_hex


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
        scroll = ScrollView(size_hint=(1, 0.8))
        self.grid = GridLayout(
            cols=2,
            spacing=10,
            size_hint_y=None,
            padding=[10, 10, 10, 10]
        )
        self.grid.bind(minimum_height=self.grid.setter('height'))
        scroll.add_widget(self.grid)
        layout.add_widget(scroll)

        # Back button
        back_btn = Button(text='Back to Options', size_hint=(1, 0.1))
        back_btn.bind(on_release=self.go_back)
        layout.add_widget(back_btn)

        self.add_widget(layout)

        # Fetch data from Django
        self.load_users()

    def load_users(self):
        url = "http://127.0.0.1:8000/api/users/"  # Django user list endpoint
        try:
            response = requests.get(url)
            response.raise_for_status()  # Raise HTTPError for bad status
            users = response.json()

            self.grid.clear_widgets()

            # Header row
            self.grid.add_widget(self.create_label("Username", bold=True))
            self.grid.add_widget(self.create_label("User ID", bold=True))

            for user in users:
                self.grid.add_widget(self.create_label(user['username']))
                self.grid.add_widget(self.create_label(str(user['id'])))

        except requests.exceptions.RequestException as e:
            self.grid.clear_widgets()
            self.grid.add_widget(self.create_label("Error loading leaderboard"))
            print("Request Error:", e)

    def create_label(self, text, bold=False):
        return Label(
            text=text,
            bold=bold,
            color=get_color_from_hex('#FFD700'),
            size_hint_y=None,
            height=40,
            padding=(10, 10)
        )

    def go_back(self, *args):
        self.manager.current = 'options'
