import numpy as np
from . import VMem as VMem
from . import ComputationUnit as ComputationUnit

'''
    - self.inputChannelIdx is the corresponding channels of the input feature map that are being processed currently.
      Assuming the # input channels <= 1024 for now. Ignoring this variable.

    - self.strides is the strides
    - self.kernel_num is the depth of the kernel. Since PE may not be enough, depth of the kernel might need to be split to differnt rounds of computation.
    - self.kernel_size is the dimension of the kernel
    - self.outputShape
    - self.startId is the ID of the kernel to start from. (0 <= self.startId <= self.kernel_num)
    - self.position is the position of the entire kernel (in grid space).
    - self.curIteration is the current iteration. The grid is (self.curIteration // kernel width, self.curIteration % kernel height) in the kernel coordinate
      A total of (kernel width) * (kernel height) iterations are required.
    - self.hasNext indicates whether the scanning is finished
'''
class KernelPostion():
    def __init__(self, strides: int, kernel_num: int, kernel_size: tuple, output_shape: tuple):
        self.strides = strides
        self.kernelNum = kernel_num
        self.kernelSize = kernel_size
        self.outputShape = output_shape

        self.startId = 0
        self.position = np.array([0, 0])
        self.curIteration = 0

        self.done = False

    def next(self, max_kernel_usage: int) -> tuple:
        '''
        1. check self.startId+max_kernel_usage, if >= self.kernelNum, reset and incrase self.position
        2. if incrase self.position, check if valid within self.outputShape, if not, reset and increase curIteration
        3. if increase self.curIteration, check if last iteration
        4. if last iteration, DO SOMETHING
        5. return the
            - kernelIDs to use
            - the global position of the kernel
            - the curIteration trasformed into coordinate in kernel space
        '''
        max_kernel_usage = min(max_kernel_usage, self.kernelNum-self.startId)
        kernelIDs = np.array(range(self.startId, self.startId+max_kernel_usage))
        globalPos = tuple(self.position)
        kernelPos = tuple([self.curIteration//self.kernelSize[1], self.curIteration%self.kernelSize[0]])

        self.startId += max_kernel_usage
        if self.startId >= self.kernelNum:
            self.startId = 0
            if self.position[1] >= self.outputShape[1]-1:
                self.position[1] = 0
                if self.position[0] >= self.outputShape[0]-1:
                    self.position[0] = 0
                    if self.curIteration >= self.kernelSize[0]*self.kernelSize[1]-1:
                        self.done = True
                    else:
                        self.curIteration += 1
                else:
                    self.position[0] += 1
            else:
                self.position[1] += 1

        return (kernelIDs, globalPos, kernelPos)

    def hasNext(self):
        return not self.done

    def __str__(self):
        return 'kernel pos {}/{}, iteration {}/{}, kernelID {}/{}'.format(tuple(self.position),
                                                                          (self.outputShape[0]-1, self.outputShape[1]-1),
                                                                          self.curIteration,
                                                                          self.kernelSize[0]*self.kernelSize[1]-1,
                                                                          self.startId,
                                                                          self.kernelNum-1)

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

    def sendData(self, max_multiplications: int) -> tuple:
        '''
      ✓ 1. make sure the buffer contains data
      ✓ 2. prepare the data for the next round
      ✓ 3. send it
        '''
        self._check()
        return self._nextRound(max_multiplications)

    def resetKernelPosition(self, strides: int, kernel_num: int, kernel_size: tuple, output_shape: tuple) -> None:
        assert len(output_shape) <= 3
        if len(output_shape) == 3:
            output_shape = (output_shape[0], output_shape[1])
        assert len(output_shape) == 2
        self.krnlPos = KernelPostion(strides, kernel_num, kernel_size=kernel_size, output_shape=output_shape)

    def _check(self) -> None:
        '''
      ✓ 1. reload when buffer is empty
        2. reload when current layer all computed
        '''
        if self.fmBuffer is None or self.krnlBuffer is None:
            self._read()
        else:
            if not self.krnlPos.hasNext():
                pass
            raise NotImplementedError

    def _read(self) -> None:
        '''
      ✓ 1. mem read
        '''
        assert self.vMem is not None
        self.vMem.send(self)

    def _nextRound(self, max_multiplications: int) -> tuple:
        '''
      ✓ 1. step self.krnlPos forward
            - the number of kernels to use for a particular channel is max_multiplications/input_channel_size
      ✓ 2. return the next batch of data
        '''
        kernelIDs, globalPos, kernelPos = self.krnlPos.next(max_multiplications//self.fmBuffer.shape[2])

        kernelWeights = self.krnlBuffer[kernelPos[0]:kernelPos[0]+1, kernelPos[1]:kernelPos[1]+1, :, kernelIDs]
        features = self.fmBuffer[globalPos[0]+kernelPos[0]:globalPos[0]+kernelPos[0]+1, globalPos[1]+kernelPos[1]:globalPos[1]+kernelPos[1]+1, :]

        return (kernelWeights, features, globalPos, kernelIDs)

    def linkVMem(self, v_mem: VMem) -> None:
        self.vMem = v_mem
