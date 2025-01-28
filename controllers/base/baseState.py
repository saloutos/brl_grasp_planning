# define base state class
class BaseState:

    def __init__(self):
        self.name = "Base"
        self.enabled = 0

    def enter(self, GP):
        self.enabled = 1

    def exit(self, GP):
        self.enabled = 0

    def execute(self, GP):
        next_state = self.name
        return next_state