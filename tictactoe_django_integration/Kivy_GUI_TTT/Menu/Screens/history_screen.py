from kivy.uix.screenmanager import Screen
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.gridlayout import GridLayout
from kivy.uix.scrollview import ScrollView
from kivy.uix.label import Label

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