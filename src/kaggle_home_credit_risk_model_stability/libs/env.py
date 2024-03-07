from pathlib import Path

class Env:
  def __init__(self, input_directory, output_directory):
    self.input_directory = Path(input_directory)
    self.output_directory = Path(output_directory)