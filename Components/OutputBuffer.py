from . import VMem as VMem
from . import ComputationUnit as ComputationUnit


'''
- self.residuals stores the required feature maps for the decoding path
- self.activeBuffer is the storage of the output feature map for the current layer
  re-initialize it when the computation loop enters the next layer might be more convenient
- self.computationalUnit references the computational unit for convenience
- self.dataReady indicates whether the data is ready to be written to the external memory
- self.end indicates whether the loop has ended (last layer or not)
'''

class OutputBuffer():
    def __init__(self) -> None:
        self.residuals = None
        self.activeBuffer = None
        self.computationalUnit = None
        self.dataReady = False
        self.end = False

    def vMemWrite(self, v_mem: VMem) -> bool:
        '''
      ✓ 1. verify the current data
            - mem write when ready
      ✓ 2. check for the status of the output
            - if the computed output is for the final layer, return True
            - otherwise, return False
        '''
        self.checkData()
        v_mem.write(self.activeBuffer)
        return self.end

    def checkData(self) -> None:
        '''
      ✓ 1. check if the current data is ready
            - if not, request data from the computational unit until the whole output is computed
        2. when the data is ready, add bias and do activation post-processing
        3. check for other required post-processing (pooling, upsampling, concatenation)
        '''
        while not self.dataReady:
            self.requestData()
        raise NotImplementedError

    def requestData(self) -> None:
        '''
      ✓ 1. request new data from the computational unit
      ✓ 2. update the status accordingly
        '''
        assert self.computationalUnit is not None
        self.dataReady, self.end = self.computationalUnit.dataWrite(self)

    def linkComputationalUnit(self, computational_unit: ComputationUnit) -> None:
        self.computationalUnit = computational_unit
