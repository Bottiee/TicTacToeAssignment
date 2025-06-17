from kivy.uix.screenmanager import Screen
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label

class OptionsScreen(Screen):
    def __init__(self, menu_manager, **kwargs):
        super().__init__(**kwargs)
        self.menu_manager = menu_manager

        layout = BoxLayout(orientation='vertical', padding=20, spacing=20)
        title = Label(text="Options Menu", font_size=32, size_hint=(1, 0.2))
        layout.add_widget(title)

        leaderboard_btn = Button(text='View Leaderboard', size_hint=(1, 0.1))
        leaderboard_btn.bind(on_release=self.goto_leaderboard)
        layout.add_widget(leaderboard_btn)

        btn_history = Button(text="View Game History", size_hint=(1, 0.2))
        btn_tile_size = Button(text="Change Tile Size", size_hint=(1, 0.2))

        # Create CPU toggle button with initial text and color
        cpu_state = self.menu_manager.is_cpu_enabled()
        self.btn_toggle_cpu = Button(
            text=f"CPU Mode: {'ON' if cpu_state else 'OFF'}",
            background_color=(0, 1, 0, 1) if cpu_state else (1, 0, 0, 1),
            size_hint=(1, 0.2)
        )
        self.btn_toggle_cpu.bind(on_release=self.toggle_cpu_mode)

        btn_back = Button(text="Return to Main Menu", size_hint=(1, 0.2))

        btn_history.bind(on_release=lambda _: setattr(self.manager, 'current', 'history'))
        btn_tile_size.bind(on_release=lambda _: setattr(self.manager, 'current', 'tile_size'))
        btn_back.bind(on_release=lambda _: setattr(self.manager, 'current', 'main'))

        layout.add_widget(btn_history)
        layout.add_widget(btn_tile_size)
        layout.add_widget(self.btn_toggle_cpu)
        layout.add_widget(btn_back)

        self.add_widget(layout)

    def toggle_cpu_mode(self, instance):
        current = self.menu_manager.is_cpu_enabled()
        new_state = not current
        self.menu_manager.set_cpu_enabled(new_state)
        # Update button text and background color based on new state
        instance.text = f"CPU Mode: {'ON' if new_state else 'OFF'}"
        instance.background_color = (0, 1, 0, 1) if new_state else (1, 0, 0, 1)

    def goto_leaderboard(self, instance):
        self.manager.current = 'leaderboard'