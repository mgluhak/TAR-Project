from progress.progress_base import ProgressBar


class SilentProgressBar(ProgressBar):
    def show_update(self):
        pass
