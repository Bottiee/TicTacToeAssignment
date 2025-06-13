from kivy.app import App
from kivy.uix.screenmanager import ScreenManager
from kivy.core.window import Window
from Kivy_GUI_TTT.Menu.Screens.main_screen import MainMenu
from Kivy_GUI_TTT.Menu.Menu_general import MenuManager
from Kivy_GUI_TTT.Menu.Screens.options_screen import OptionsScreen
from Kivy_GUI_TTT.Menu.Screens.history_screen import HistoryScreen
from Kivy_GUI_TTT.Menu.Screens.tile_size_screen import TileSizeScreen
from Kivy_GUI_TTT.Menu.Screens.game_screen import GameScreen
from Kivy_GUI_TTT.Decorations.kivy_shader_silk import SilkWidget
from kivy.uix.floatlayout import FloatLayout

Window.size = (700, 900)

class RootWidget(FloatLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Set initial size and position
        initial_pos = (-1, -1)
        initial_size = Window.size

        self.background = SilkWidget(size=initial_size, pos=initial_pos)
        self.add_widget(self.background)

        # Add shader background
        self.background = SilkWidget(size=Window.size, pos=(0, 0))
        self.add_widget(self.background)

        # Create ScreenManager and add all screens
        self.sm = ScreenManager()
        self.menu_manager = MenuManager()
        self.sm.add_widget(MainMenu(name='main', menu_manager=self.menu_manager))
        self.sm.add_widget(GameScreen(name='game', menu_manager=self.menu_manager))
        self.sm.add_widget(OptionsScreen(name='options', menu_manager=self.menu_manager))
        self.sm.add_widget(HistoryScreen(name='history', menu_manager=self.menu_manager))
        self.sm.add_widget(TileSizeScreen(name='tile_size', menu_manager=self.menu_manager))

        # Add ScreenManager on top of shader background
        self.add_widget(self.sm)

        # Bind window resize to update shader background size
        Window.bind(on_resize=self._on_window_resize)

    # noinspection PyUnusedLocal
    def _on_window_resize(self, window, width, height):
        self.background.pos = (-1, -1)
        self.background.size = (width, height)


class TicTacToeApp(App):
    def build(self):
        return RootWidget()

    def on_stop(self):
        pass

if __name__ == '__main__':
    TicTacToeApp().run()