# Kivy_GUI_TTT/Menu/Screens/reg_screen.py
from kivy.uix.screenmanager import Screen
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
import threading
import requests
from kivy.clock import mainthread

class RegisterUserScreen(Screen):
    def __init__(self, menu_manager, **kwargs):
        super().__init__(**kwargs)
        self.menu_manager = menu_manager

        layout = BoxLayout(orientation='vertical', spacing=10, padding=40)

        layout.add_widget(Label(text='Register User', font_size=24))

        self.username_input = TextInput(hint_text='Username', multiline=False)
        layout.add_widget(self.username_input)

        self.password_input = TextInput(hint_text='Password', multiline=False, password=True)
        layout.add_widget(self.password_input)

        register_btn = Button(text='Register')
        register_btn.bind(on_release=self.register_user)
        layout.add_widget(register_btn)

        back_btn = Button(text='Back')
        back_btn.bind(on_release=self.go_back)
        layout.add_widget(back_btn)

        self.add_widget(layout)

        self.status_label = Label(text='')
        layout.add_widget(self.status_label)

    def register_user(self, instance):
        username = self.username_input.text.strip()
        password = self.password_input.text.strip()

        if not username or not password:
            self.update_status("Username and password required")
            return

        # Run the request in a background thread to avoid blocking UI
        threading.Thread(target=self._send_registration_request, args=(username, password), daemon=True).start()

    def _send_registration_request(self, username, password):
        url = "http://127.0.0.1:8000/api/register/"  # Your Django registration endpoint
        payload = {
            "username": username,
            "password": password
        }
        try:
            response = requests.post(url, json=payload)
            if response.status_code == 201:
                self.update_status("Registration successful!")
            else:
                # Extract error message from Django response
                try:
                    err = response.json().get('error', response.text)
                except Exception:
                    err = response.text
                self.update_status(f"Registration failed: {err}")
        except requests.RequestException as e:
            self.update_status(f"Request error: {e}")

    @mainthread
    def update_status(self, message):
        self.status_label.text = message

    def go_back(self, instance):
        self.manager.current = 'login'