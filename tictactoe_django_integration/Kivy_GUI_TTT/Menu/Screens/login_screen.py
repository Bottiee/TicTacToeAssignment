import requests
import json
import threading

from kivy.uix.screenmanager import Screen
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.utils import get_color_from_hex
from kivy.app import App
from kivy.clock import mainthread


class LoginScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = 'login'

        layout = BoxLayout(orientation='vertical', padding=30, spacing=20)

        title = Label(
            text='User Login',
            font_size='36sp',
            size_hint=(1, 0.2),
            color=get_color_from_hex('#FFD700')
        )
        layout.add_widget(title)

        self.username_input = TextInput(
            hint_text='Username',
            font_size='20sp',
            size_hint=(1, 0.1),
            padding=[10, 10, 10, 10],
            multiline=False
        )
        layout.add_widget(self.username_input)

        self.password_input = TextInput(
            hint_text='Password',
            font_size='20sp',
            size_hint=(1, 0.1),
            padding=[10, 10, 10, 10],
            multiline=False,
            password=True
        )
        layout.add_widget(self.password_input)

        login_btn = Button(
            text='Login',
            font_size='24sp',
            size_hint=(1, 0.15),
            background_color=get_color_from_hex('#4CAF50'),
            color=(1, 1, 1, 1)
        )
        login_btn.bind(on_release=self.trigger_login)
        layout.add_widget(login_btn)

        register_btn = Button(
            text='Register',
            font_size='24sp',
            size_hint=(1, 0.15),
            background_color=get_color_from_hex('#2196F3'),
            color=(1, 1, 1, 1)
        )
        register_btn.bind(on_release=self.trigger_register)
        layout.add_widget(register_btn)

        self.status_label = Label(
            text='',
            font_size='18sp',
            size_hint=(1, 0.1),
            color=get_color_from_hex('#FF0000')
        )
        layout.add_widget(self.status_label)

        self.add_widget(layout)

    def trigger_login(self, *args):
        username = self.username_input.text.strip()
        password = self.password_input.text.strip()
        if not username or not password:
            self.update_status("Please enter both username and password.", is_error=True)
            return

        self.update_status("Logging in...", is_error=False)
        threading.Thread(target=self._perform_login, args=(username, password)).start()

    def _perform_login(self, username, password):
        url = "http://127.0.0.1:8000/api-token-auth/"
        try:
            response = requests.post(url, data={'username': username, 'password': password})

            print(f"\n--- Django Login Response ---")
            print(f"Status Code: {response.status_code}")
            print(f"Response Body: {response.text}")
            print(f"----------------------------\n")

            response.raise_for_status()
            token_data = response.json()
            auth_token = token_data.get('token')

            if auth_token:
                app = App.get_running_app()
                app.user_token = auth_token
                app.current_username = username

                self.update_status("Login successful!", is_error=False)
                self.go_to_options_screen()
            else:
                self.update_status("Login failed: No token received.", is_error=True)

        except requests.exceptions.ConnectionError:
            self.update_status("Connection Error: Could not connect to Django server.", is_error=True)
        except requests.exceptions.HTTPError as e:
            error_msg = "Login failed."
            if e.response and e.response.text:
                try:
                    error_json = e.response.json()
                    errors = error_json.get('non_field_errors', error_json.get('detail', [error_msg]))
                    if isinstance(errors, list) and errors:
                        error_msg = errors[0]
                    else:
                        error_msg = str(errors)
                except (json.JSONDecodeError, KeyError):
                    error_msg = e.response.text
            self.update_status(f"Login Failed ({e.response.status_code}): {error_msg}", is_error=True)
        except json.JSONDecodeError:
            self.update_status("Login Error: Invalid JSON response from server.", is_error=True)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.update_status(f"An unexpected error occurred: {e}", is_error=True)

    def trigger_register(self, *args):
        self.manager.current = 'register'

    @mainthread
    def update_status(self, message, is_error=True):
        self.status_label.text = message
        self.status_label.color = get_color_from_hex('#FF0000') if is_error else get_color_from_hex('#00FF00')

    @mainthread
    def go_to_options_screen(self):
        self.manager.current = 'options'
