from kivy.uix.screenmanager import Screen
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.gridlayout import GridLayout
from Kivy_GUI_TTT.Logic.Conditionals import check_win_condition, check_draw_condition
from Kivy_GUI_TTT.Logic.Computer_logic import get_cpu_move
from kivy.clock import Clock
from kivy.uix.label import Label

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

