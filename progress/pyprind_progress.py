from progress.progress_base import ProgressBar
from pyprind import ProgBar


class PyprindProgressBar(ProgressBar):

    def __init__(self):
        super().__init__()
        self.pBar = None

    def initialize(self, max_value, start=0, step=1, print_step=1):
        super().initialize(max_value, start, step, print_step)
        self.pBar = ProgBar(int(max_value / print_step))

    def show_update(self):
        self.pBar.update(self.step)
