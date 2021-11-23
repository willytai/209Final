from . import VMem as VMem
from . import ComputationUnit as ComputationUnit


'''
    - self.computationalUnit references the computational unit for convenience
'''

class OutputBuffer():
    def __init__(self) -> None:
        self.computationalUnit = None

    def vMemWrite(self, v_mem: VMem) -> bool:
        '''
        1. check if all the elements of the current output are computed
        2. check for the status of the output
            - if the computed output is for the final layer, return True
            - otherwise, return False
        '''
        assert self.computationalUnit is not None
        raise NotImplementedError

    def linkComputationalUnit(self, computational_unit: ComputationUnit) -> None:
        self.computationalUnit = computational_unit
