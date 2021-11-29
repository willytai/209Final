from . import VMem as VMem
from . import ComputationUnit as ComputationUnit

'''
    - self.inputChannelIdx is the corresponding channels of the input feature map that are being processed currently.
      Assuming the # input channels <= 1024 for now. Ignoring this variable.

    - self.curIteration is the current iteration represented by the coordinate of the kernel (in kernel space).
      A total of (kernel width) * (kernel height) iterations are required.
    - self.strides is the strides
    - self.kernel_num is the depth of the kernel. Since PE may not be enough, depth of the kernel might need to be split to differnt rounds of computation.
    - self.position is the position of the entire kernel (in grid space).
    - self.startId is the ID of the kernel to start from.
'''
class KernelPostion():
    def __init__(self, strides: int, kernel_num: int):
        self.reset(strides, kernel_num)

    def reset(self, strides: int, kernel_num: int) -> None:
        self.strides = strides
        self.kernel_num = kernel_num

        self.position = (0, 0)
        self.curIteration = (0, 0)
        self.startId = 0

    def next(self) -> None:
        '''
        might need some addtional parameters
        '''
        raise NotImplementedError

'''
    - self.vMem references the external memory for convenience
    - self.fmBuffer is the feature map buffer
    - self.krnlBuffer is the kernel buffer
    - self.krnlPos is the information of the current position of the kernels and the kernels that have already processed, iteration index is also stored
'''
class InputBuffer():
    def __init__(self):
        self.vMem = None
        self.fmBuffer = None
        self.krnlBuffer = None
        self.krnlPos = None

    def sendData(self) -> tuple:
        '''
      ✓ 1. make sure the buffer contains data
      ✓ 2. prepare the data for the next round
      ✓ 3. send it
        '''
        self._check()
        return self._nextRound()

    def _check(self) -> None:
        '''
      ✓ 1. reload when buffer is empty
        '''
        if self.fmBuffer is None or self.krnlBuffer is None:
            self._read()

    def _read(self) -> None:
        '''
        1. read the next conv layer from VMem and load feature map and kernel
        2. send instruction to the output buffer for buffer initialization
        3. reset self.krnlPos
        4. do the padding here
        '''
        assert self.vMem is not None
        self.vMem.read(self)
        raise NotImplementedError

    def _nextRound(self) -> tuple:
        '''
      ✓ 1. step self.krnlPos forward
        2. return the next batch of data
        '''
        self.krnlPos.next()
        raise NotImplementedError

    def linkVMem(self, v_mem: VMem) -> None:
        self.vMem = v_mem
