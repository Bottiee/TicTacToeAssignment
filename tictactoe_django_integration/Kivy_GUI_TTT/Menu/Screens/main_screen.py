from kivy.uix.screenmanager import Screen
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from tictactoe_django_integration.Kivy_GUI_TTT.Menu.Screens.quit import quit_game


class MainMenu(Screen):
    def __init__(self, menu_manager, **kwargs):
        super().__init__(**kwargs)
        self.menu_manager = menu_manager

        layout = BoxLayout(orientation='vertical', spacing=20, padding=40)
        layout.add_widget(Label(text="Tic Tac Toe", font_size=32))

        btn_start = Button(text="Start Game", size_hint=(1, 0.2))
        btn_options = Button(text="Options", size_hint=(1, 0.2))
        btn_register = Button(text="Register User", size_hint=(1, 0.2))  # New button
        btn_quit = Button(text="Quit", size_hint=(1, 0.2))

        btn_start.bind(on_release=self.start_game)
        btn_options.bind(on_release=self.open_options)
        btn_register.bind(on_release=self.open_register)  # Bind new button
        btn_quit.bind(on_release=lambda *_: quit_game())

        layout.add_widget(btn_start)
        layout.add_widget(btn_options)
        layout.add_widget(btn_register)  # Add button to layout
        layout.add_widget(btn_quit)

        self.add_widget(layout)

    def start_game(self, instance):
        self.manager.current = 'game'

    def open_options(self, instance):
        self.manager.current = 'options'

    def open_register(self, instance):
        self.manager.current = 'register'