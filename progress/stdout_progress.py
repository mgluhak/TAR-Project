from progress.progress_base import ProgressBar


class StdoutProgressBar(ProgressBar):
    def show_update(self):
        print("iteration = " + str(self.counter) + "/" + str(self.max_value))
