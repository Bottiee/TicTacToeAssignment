from kivy.uix.screenmanager import Screen
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.gridlayout import GridLayout
from Kivy_GUI_TTT.Logic.Conditionals import check_win_condition
from Kivy_GUI_TTT.Logic.Conditionals import check_draw_condition
from kivy.clock import Clock
from kivy.uix.label import Label
from Kivy_GUI_TTT.Logic.Computer_logic import ComputerLogic

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

    def cpu_move(self, *args):
        if self.ai:
            move = self.ai.get_ai_move(self.board_state)
            if move:
                r, c = move
                # Only make move if cell is empty (just to be safe)
                if self.board_state[r][c] == '':
                    self.board_state[r][c] = 'O'  # AI is always 'O'
                    self.buttons[r][c].text = 'O'

                    if check_win_condition(self.board_state, 'O'):
                        self.turn_label.text = "O wins!"
                        self.menu_manager.record_game_result('O')
                        self.disable_board()
                        return

                    if check_draw_condition(self.board_state):
                        self.turn_label.text = "It's a draw!"
                        self.disable_board()
                        self.menu_manager.record_draw()
                        return

                    # Switch turn back to 'X' after AI move
                    self.current_turn = 'X'
                    self.turn_label.text = f"Turn: {self.current_turn}"
            else:
                print("AI did not return a valid move")

    def init_game(self):
        # Reset state
        self.board_size = self.menu_manager.get_tile_size()
        self.cpu_enabled = self.menu_manager.is_cpu_enabled()
        self.current_turn = 'X'
        self.board_state = [['' for _ in range(self.board_size)] for _ in range(self.board_size)]

        # Clear existing widgets from the screen
        self.clear_widgets()

        # Initialize layout and turn label ONCE
        self.layout = BoxLayout(orientation='vertical', padding=15, spacing=15)
        self.turn_label = Label(text=f"Turn: {self.current_turn}", font_size=24, size_hint=(1, 0.1))
        self.layout.add_widget(self.turn_label)

        if self.cpu_enabled:
            self.ai = ComputerLogic(board_size=self.board_size)
        else:
            self.ai = None

        # Show CPU info if enabled
        if self.cpu_enabled:
            cpu_info_label = Label(
                text="CPU Mode: ON",
                color=(1, 0.5, 0, 1),  # Orange color
                font_size=18,
                size_hint=(1, 0.1)
            )
            self.layout.add_widget(cpu_info_label)

        # Initialize the game grid
        self.grid = GridLayout(cols=self.board_size, rows=self.board_size, spacing=5, size_hint=(1, 0.75))
        self.buttons = [] # Ensure buttons list is cleared for new cdgame

        for row in range(self.board_size):
            btn_row = []
            for col in range(self.board_size):
                btn = Button(text='', font_size=40)
                btn.bind(on_release=lambda inst, r=row, c=col: self.tile_clicked(r, c))
                self.grid.add_widget(btn)
                btn_row.append(btn)
            self.buttons.append(btn_row)
        self.layout.add_widget(self.grid)

        # Add control buttons
        btn_layout = BoxLayout(size_hint=(1, 0.15), spacing=10)
        reset_btn = Button(text='Reset Board')
        reset_btn.bind(on_release=self.reset_board)
        back_btn = Button(text='Back to Main Menu')
        back_btn.bind(on_release=self.go_back)
        btn_layout.add_widget(reset_btn)
        btn_layout.add_widget(back_btn)

        self.layout.add_widget(btn_layout)
        self.add_widget(self.layout) # Add the main layout to the screen

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
    def reset_board(self, instance):
        self.init_game()

    def disable_board(self):
        for row in self.buttons:
            for btn in row:
                btn.disabled = True

    # noinspection PyUnusedLocal
    def go_back(self, instance):
        self.manager.current = 'main'
