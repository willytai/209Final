from . import VMem as VMem
from . import ComputationUnit as ComputationUnit


'''
- self.residuals stores the required feature maps for the decoding path
- self.activeBuffer is the storage of the output feature map for the current layer
  re-initialize it when the computation loop enters the next layer might be more convenient
- self.computationUnit references the computation unit for convenience
- self.dataReady indicates whether the data is ready to be written to the external memory
- self.end indicates whether the loop has ended (last layer or not)
'''

class OutputBuffer():
    def __init__(self) -> None:
        self.residuals = None
        self.activeBuffer = None
        self.computationUnit = None
        self.dataReady = False
        self.end = False

    def vMemWrite(self, v_mem: VMem) -> bool:
        '''
      ✓ 1. verify the current data
      ✓ 2. post-processing
      ✓ 3. mem write
      ✓ 4. reset the active buffer
      ✓ 5. check for the status of the output
            - if the computed output is for the final layer, return True
            - otherwise, return False
        '''
        self.checkData()
        self.postProcess()
        v_mem.write(self.activeBuffer)
        self.activeBuffer = None
        return self.end

    def checkData(self) -> None:
        '''
      ✓ 1. check if the current data is ready
            - if not, request data from the computation unit until the whole output is computed
        '''
        while not self.dataReady:
            self.requestData()
        raise NotImplementedError

    def requestData(self) -> None:
        '''
      ✓ 1. request new data from the computation unit
            - computation
            - data write
      ✓ 2. update the status accordingly
        '''
        assert self.computationUnit is not None
        self.computationUnit.computeNextRound()
        self.dataReady, self.end = self.computationUnit.dataWrite(self)

    def postProcess(self) -> None:
        '''
        1. add bias and do activation post-processing
        2. check for other required post-processing (pooling, upsampling, concatenation)
        '''
        raise NotImplementedError

    def linkComputationUnit(self, computation_unit: ComputationUnit) -> None:
        self.computationUnit = computation_unit
