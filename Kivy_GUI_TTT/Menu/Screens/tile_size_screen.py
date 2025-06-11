from kivy.uix.screenmanager import Screen
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label

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
