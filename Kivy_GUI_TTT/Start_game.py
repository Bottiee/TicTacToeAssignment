from kivy.app import App
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.core.window import Window
from Kivy_GUI_TTT.Menu.Screens.main_screen import MainMenu
from Kivy_GUI_TTT.Menu.Menu_general import MenuManager
from Kivy_GUI_TTT.Menu.Screens.options_screen import OptionsScreen
from Kivy_GUI_TTT.Menu.Screens.history_screen import HistoryScreen
from Kivy_GUI_TTT.Menu.Screens.tile_size_screen import TileSizeScreen
from Kivy_GUI_TTT.Menu.Screens.game_screen import GameScreen

Window.size = (400, 600)


def quit_game():
    App.get_running_app().stop()

class TicTacToeApp(App):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sm = None
        self.menu_manager = MenuManager()

    def build(self):
        self.menu_manager = MenuManager()
        self.sm = ScreenManager()
        self.sm.add_widget(MainMenu(name='main', menu_manager=self.menu_manager))
        self.sm.add_widget(GameScreen(name='game', menu_manager=self.menu_manager))   # ADD THIS LINE
        self.sm.add_widget(OptionsScreen(name='options', menu_manager=self.menu_manager))
        self.sm.add_widget(HistoryScreen(name='history', menu_manager=self.menu_manager))
        self.sm.add_widget(TileSizeScreen(name='tile_size', menu_manager=self.menu_manager))
        return self.sm

    def on_stop(self):
        self.menu_manager.close()

if __name__ == '__main__':
    TicTacToeApp().run()