# Kivy_GUI_TTT/Start_game.py
from kivy.app import App
from kivy.uix.screenmanager import ScreenManager
from kivy.core.window import Window
from kivy.uix.floatlayout import FloatLayout

from tictactoe_django_integration.Kivy_GUI_TTT.Menu.Screens.main_screen import MainMenu
from tictactoe_django_integration.Kivy_GUI_TTT.Menu.Menu_general import MenuManager
from tictactoe_django_integration.Kivy_GUI_TTT.Menu.Screens.options_screen import OptionsScreen
from tictactoe_django_integration.Kivy_GUI_TTT.Menu.Screens.history_screen import HistoryScreen
from tictactoe_django_integration.Kivy_GUI_TTT.Menu.Screens.tile_size_screen import TileSizeScreen
from tictactoe_django_integration.Kivy_GUI_TTT.Menu.Screens.game_screen import GameScreen
from tictactoe_django_integration.Kivy_GUI_TTT.Menu.Screens.leaderboard_screen import LeaderboardScreen
from tictactoe_django_integration.Kivy_GUI_TTT.Menu.Screens.reg_screen import RegisterUserScreen
from tictactoe_django_integration.Kivy_GUI_TTT.Decorations.kivy_shader_silk import SilkWidget
from tictactoe_django_integration.Kivy_GUI_TTT.Menu.Screens.login_screen import LoginScreen

Window.size = (700, 900)

class RootWidget(FloatLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Background shader
        self.background = SilkWidget(size=Window.size, pos=(-1, -1))
        self.add_widget(self.background)

        # ScreenManager setup
        self.sm = ScreenManager()

        # Menu manager initialized AFTER screen manager is ready
        self.menu_manager = MenuManager(screen_manager=self.sm)

        # Screens registered
        self.sm.add_widget(LoginScreen(name='login'))
        self.sm.add_widget(MainMenu(name='main', menu_manager=self.menu_manager))
        self.sm.add_widget(GameScreen(name='game', menu_manager=self.menu_manager))
        self.sm.add_widget(OptionsScreen(name='options', menu_manager=self.menu_manager))
        self.sm.add_widget(HistoryScreen(name='history', menu_manager=self.menu_manager))
        self.sm.add_widget(TileSizeScreen(name='tile_size', menu_manager=self.menu_manager))
        self.sm.add_widget(LeaderboardScreen(name='leaderboard'))
        self.sm.add_widget(RegisterUserScreen(name='register', menu_manager=self.menu_manager))
        self.sm.current = 'login'
        self.add_widget(self.sm)

        # Responsive background
        Window.bind(on_resize=self._on_window_resize)

    def _on_window_resize(self, window, width, height):
        self.background.pos = (-1, -1)
        self.background.size = (width, height)

class TicTacToeApp(App):
    def build(self):
        return RootWidget()

if __name__ == '__main__':
    TicTacToeApp().run()
