from . import OutputBuffer as OutputBuffer

'''
    A PE is capable of doing 32 multiplications
'''
class PE():
    def __init__(self):
        pass

'''
    1. Contains the PE array
    2. Channel based parallelism
    - self.inputChannelIdx is the corresponding channels of the input feature map that are being processed currently. (np.array)
    - self.outputChannelIdx is the corresponding channels of the output feature that will be computed. (np.array)
    - self.currentRound is the current round represented by the coordinate of the kernel. A total of (kernel width) * (kernel height) rounds are required. (tuple)
'''
class ComputationUnit():
    def __init__(self, pe_array_size: int) -> None:
        self.PEArray = list()
        self.inputChannelIdx = None
        self.outputChannelIdx = None
        self.currentRound = None

    def computeNextRound(self) -> None:
        '''
        1. data fetch
        2. record self.inputChannelIdx, self.outputChannelIdx, self.currentRound
        3. assign task to PE
        4. acccumulate values for corresponding output channels
        '''
        raise NotImplementedError

    def dataWrite(self, output_buffer: OutputBuffer) -> tuple:
        raise NotImplementedError
