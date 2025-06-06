from kivy.app import App
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.core.window import Window
from kivy.uix.scrollview import ScrollView
from kivy.uix.gridlayout import GridLayout
from kivy.clock import Clock
from Kivy_GUI_TTT.Menu.Menu_general import MenuManager
from Kivy_GUI_TTT.Logic.Conditionals import check_win_condition, check_draw_condition
from Kivy_GUI_TTT.Logic.Computer_logic import get_cpu_move

Window.size = (400, 600)


def quit_game():
    App.get_running_app().stop()


class MainMenu(Screen):
    def __init__(self, menu_manager, **kwargs):
        super().__init__(**kwargs)
        self.menu_manager = menu_manager

        layout = BoxLayout(orientation='vertical', spacing=20, padding=40)
        layout.add_widget(Label(text="Tic Tac Toe", font_size=32))

        btn_start = Button(text="Start Game", size_hint=(1, 0.2))
        btn_options = Button(text="Options", size_hint=(1, 0.2))
        btn_quit = Button(text="Quit", size_hint=(1, 0.2))
        btn_start.bind(on_release=self.start_game)
        btn_options.bind(on_release=self.open_options)
        btn_quit.bind(on_release=lambda *_: quit_game())

        layout.add_widget(btn_start)
        layout.add_widget(btn_options)
        layout.add_widget(btn_quit)

        self.add_widget(layout)

    # noinspection PyUnusedLocal
    def start_game(self, instance):
        self.manager.current = 'game'

    # noinspection PyUnusedLocal
    def open_options(self, instance):
        self.manager.current = 'options'


class OptionsScreen(Screen):
    def __init__(self, menu_manager, **kwargs):
        super().__init__(**kwargs)
        self.menu_manager = menu_manager

        layout = BoxLayout(orientation='vertical', padding=20, spacing=20)
        title = Label(text="Options Menu", font_size=32, size_hint=(1, 0.2))
        layout.add_widget(title)

        btn_history = Button(text="View Game History", size_hint=(1, 0.2))
        btn_tile_size = Button(text="Change Tile Size", size_hint=(1, 0.2))
        btn_toggle_cpu = Button(text="Toggle CPU Mode", size_hint=(1, 0.2))
        btn_back = Button(text="Return to Main Menu", size_hint=(1, 0.2))
        btn_history.bind(on_release=lambda _: setattr(self.manager, 'current', 'history'))
        btn_tile_size.bind(on_release=lambda _: setattr(self.manager, 'current', 'tile_size'))
        btn_toggle_cpu.bind(on_release=self.toggle_cpu_mode)
        btn_back.bind(on_release=lambda _: setattr(self.manager, 'current', 'main'))

        layout.add_widget(btn_history)
        layout.add_widget(btn_tile_size)
        layout.add_widget(btn_toggle_cpu)
        layout.add_widget(btn_back)

        self.add_widget(layout)

    def toggle_cpu_mode(self, instance):
        current = self.menu_manager.is_cpu_enabled()
        self.menu_manager.set_cpu_enabled(not current)
        instance.text = f"CPU Mode: {'ON' if not current else 'OFF'}"


class TileSizeScreen(Screen):
    def __init__(self, menu_manager, **kwargs):
        super().__init__(**kwargs)
        self.menu_manager = menu_manager

        layout = BoxLayout(orientation='vertical', padding=20, spacing=20)
        title = Label(text="Select Tile Size", font_size=32, size_hint=(1, 0.2))
        layout.add_widget(title)

        for size in range(3, 6):
            btn = Button(text=f"{size} x {size}", size_hint=(1, 0.15))
            btn.bind(on_release=lambda inst, s=size: self.set_tile_size(s))
            layout.add_widget(btn)

        btn_back = Button(text="Back to Options Menu", size_hint=(1, 0.15))
        btn_back.bind(on_release=self.go_back)
        layout.add_widget(btn_back)

        self.add_widget(layout)

    def set_tile_size(self, size):
        self.menu_manager.set_tile_size(size)
        self.manager.current = 'options'

    # noinspection PyUnusedLocal
    def go_back(self, instance):
        self.manager.current = 'options'


class HistoryScreen(Screen):
    def __init__(self, menu_manager, **kwargs):
        super().__init__(**kwargs)
        self.menu_manager = menu_manager
        self.bind(on_enter=lambda *_: self.populate_history())

        layout = BoxLayout(orientation='vertical', padding=20, spacing=20)
        title = Label(text="Game History", font_size=32, size_hint=(1, 0.1))
        layout.add_widget(title)

        self.scroll = ScrollView(size_hint=(1, 0.7))
        self.grid = GridLayout(cols=2, size_hint_y=None, spacing=10, padding=10)
        self.grid.bind(minimum_height=self.grid.setter('height'))
        self.scroll.add_widget(self.grid)

        layout.add_widget(self.scroll)
        btn_clear = Button(text="Clear Game History", size_hint=(1, 0.1))
        btn_clear.bind(on_release=self.clear_history)
        layout.add_widget(btn_clear)
        btn_back = Button(text="Back to Options Menu", size_hint=(1, 0.15))
        btn_back.bind(on_release=self.go_back)
        layout.add_widget(btn_back)

        self.add_widget(layout)
        self.populate_history()

    def populate_history(self):
        self.grid.clear_widgets()
        stats = self.menu_manager.get_history_dict()
        if not stats or self.menu_manager.history_storage.total_games == 0:
            self.grid.add_widget(Label(text="No history available", font_size=20, size_hint_y=None, height=30))
            self.grid.add_widget(Label(text="", size_hint_y=None, height=30))
            return
        for key, value in stats.items():
            self.grid.add_widget(Label(text=f"{key}:", font_size=20, size_hint_y=None, height=30))
            self.grid.add_widget(Label(text=str(value), font_size=20, size_hint_y=None, height=30))

    # noinspection PyUnusedLocal
    def clear_history(self, instance):
        self.menu_manager.clear_history()
        self.populate_history()

    # noinspection PyUnusedLocal
    def go_back(self, instance):
        self.manager.current = 'options'


class GameScreen(Screen):
    def __init__(self, menu_manager, **kwargs):
        super().__init__(**kwargs)
        self.menu_manager = menu_manager

        # Declare attributes with placeholders
        self.layout = None
        self.grid = None
        self.buttons: list[list[Button]] = []
        self.turn_label = None
        self.current_turn: str = 'X'
        self.board_size: int | None = None
        self.cpu_enabled: bool | None = None
        self.board_state: list[list[str]] | None = None

        # Rebuild UI on entering screen to reflect tile size changes dynamically
        self.bind(on_enter=lambda *_: self.init_game())

    def init_game(self):
        # Reset state
        self.board_size = self.menu_manager.get_tile_size()
        self.cpu_enabled = self.menu_manager.is_cpu_enabled()
        self.current_turn = 'X'
        self.board_state = [['' for _ in range(self.board_size)] for _ in range(self.board_size)]

        # Clear and build UI
        self.clear_widgets()
        self.layout = BoxLayout(orientation='vertical', padding=10, spacing=10)
        self.turn_label = Label(text=f"Turn: {self.current_turn}", font_size=24, size_hint=(1, 0.1))
        self.layout.add_widget(self.turn_label)
        self.grid = GridLayout(cols=self.board_size, rows=self.board_size, spacing=5, size_hint=(1, 0.75))
        self.buttons = []

        for row in range(self.board_size):
            btn_row = []
            for col in range(self.board_size):
                btn = Button(text='', font_size=40)
                btn.bind(on_release=lambda inst, r=row, c=col: self.tile_clicked(r, c))
                self.grid.add_widget(btn)
                btn_row.append(btn)
            self.buttons.append(btn_row)
        self.layout.add_widget(self.grid)

        btn_layout = BoxLayout(size_hint=(1, 0.15), spacing=10)
        reset_btn = Button(text='Reset Board')
        reset_btn.bind(on_release=self.reset_board)
        back_btn = Button(text='Back to Main Menu')
        back_btn.bind(on_release=self.go_back)
        btn_layout.add_widget(reset_btn)
        btn_layout.add_widget(back_btn)

        self.layout.add_widget(btn_layout)
        self.add_widget(self.layout)

    def tile_clicked(self, row, col):
        if self.board_state[row][col] != '':
            return
        self.board_state[row][col] = self.current_turn
        self.buttons[row][col].text = self.current_turn

        if check_win_condition(self.board_state, self.current_turn):
            self.turn_label.text = f"{self.current_turn} wins!"
            self.menu_manager.record_game_result(self.current_turn)
            self.disable_board()
            return

        if check_draw_condition(self.board_state):
            self.turn_label.text = "It's a draw!"
            self.disable_board()
            self.menu_manager.record_draw()
            return

        self.switch_turn()
        if self.current_turn == 'O' and self.cpu_enabled:
            Clock.schedule_once(lambda dt: self.cpu_move(), 0.2)

    def switch_turn(self):
        self.current_turn = 'O' if self.current_turn == 'X' else 'X'
        self.turn_label.text = f"Turn: {self.current_turn}"

    # noinspection PyUnusedLocal
    def cpu_move(self, *args):
        move = get_cpu_move(self.board_state)
        if move:
            r, c = move
            self.tile_clicked(r, c)

    # noinspection PyUnusedLocal
    def reset_board(self, instance):
        self.init_game()

    def disable_board(self):
        for row in self.buttons:
            for btn in row:
                btn.disabled = True

    # noinspection PyUnusedLocal
    def go_back(self, instance):
        self.manager.current = 'main'


class TicTacToeApp(App):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sm = None # ScreenManager instance placeholder
        self.menu_manager= MenuManager()

    def build(self):
        self.sm = ScreenManager()
        self.sm.add_widget(MainMenu(name='main', menu_manager=self.menu_manager))
        self.sm.add_widget(OptionsScreen(name='options', menu_manager=self.menu_manager))
        self.sm.add_widget(TileSizeScreen(name='tile_size', menu_manager=self.menu_manager))
        self.sm.add_widget(HistoryScreen(name='history', menu_manager=self.menu_manager))
        self.sm.add_widget(GameScreen(name='game', menu_manager=self.menu_manager))
        return self.sm

    def on_stop(self):
        self.menu_manager.close()

if __name__ == '__main__':
    TicTacToeApp().run()