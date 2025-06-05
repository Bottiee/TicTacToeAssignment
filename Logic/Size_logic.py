# Size_logic.py

class SizeLogic:
    def __init__(self, initial_size=3):
        self.available_sizes = [3, 4, 5]
        self.selected_size = initial_size

    def set_size(self, size):
        if size in self.available_sizes:
            self.selected_size = size
        else:
            raise ValueError(f"Unsupported grid size: {size}")

    def get_size(self):
        return self.selected_size
