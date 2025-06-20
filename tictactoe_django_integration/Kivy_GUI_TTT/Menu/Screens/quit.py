from kivy.app import App

def quit_game():
    App.get_running_app().stop()
# Solves the circular import issue by defining the quit_game function here