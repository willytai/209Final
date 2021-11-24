from . import OutputBuffer as OutputBuffer

class PE():
    def __init__(self):
        pass

'''
    1. Contains the PE array
    2. Channel based parallelism
'''
class ComputationUnit():
    def __init__(self, pe_array_size: int) -> None:
        self.PEArray = list()

    def dataWrite(self, output_buffer: OutputBuffer) -> tuple:
        raise NotImplementedError
