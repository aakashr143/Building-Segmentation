import os
import datetime


class Logger:
    def __init__(self, model_name: str, backbone: str):
        self._root_dir = "Checkpoints"
        self._architecture = model_name + "_" + backbone
        self._timestamp = str(datetime.datetime.now()).split(".")[0].replace(":", "_")

        if not os.path.exists(os.path.join(self._root_dir, self._architecture)):
            os.makedirs(os.path.join(self._root_dir, self._architecture))

        self.folder = os.path.join(self._root_dir, self._architecture, self._timestamp)
        self.log_file = os.path.join(self.folder, "log.txt")

        os.makedirs(self.folder)

        with open(self.log_file, "w") as file:
            file.write(f"Model: {model_name}, Backbone: {backbone}\n")

    def log(self, message: str):
        with open(self.log_file, "a") as file:
            file.write(f"{message}\n")


if __name__ == "__main__":
    logger = Logger("Model", "Backbone")
    logger.log("Test message")
