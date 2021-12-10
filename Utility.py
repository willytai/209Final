import numpy as np

class Quantizer:
    __instance = None
    def __init__(self):
        if Quantizer.__instance is not None:
            raise Exception('Do not initiate this class explicitly, get the instance by calling Quantizer.getInstance() instead.')
        Quantizer.__instance = self
        self.wordLength = None
        self.lut = dict()

    def setWordLength(self, word_length: int) -> None:
        self.wordLength = word_length

        # fraction length ranges from 0 to word_length
        for fl in range(word_length):
            intLen = self.wordLength - fl
            precision = pow(2, -fl)
            valueRange = (-pow(2, intLen-1) + precision, pow(2, intLen-1) - precision)
            self.lut[fl] = valueRange

    def quantize(self, array: np.array) -> np.array:
        if self.wordLength is None:
            raise ValueError('word length not set')

        maxVal = array.max()
        minVal = array.min()
        finalFL = 0
        for fl in range(8):
            refMin, refMax = self.lut[fl]
            if refMin <= minVal and maxVal <= refMax:
                finalFL = int(fl)
            else: break

        print ('quantizing data with word length: {}, fraction length: {}'.format(self.wordLength, finalFL))

        # some tricks (float -> fixed -> float)
        array_q = array * pow(2, finalFL)
        array_q = np.round(array_q)
        array_q = array_q * pow(2, -finalFL)

        # saturate the arrays
        array_q = np.maximum(array_q, self.lut[finalFL][0])
        array_q = np.minimum(array_q, self.lut[finalFL][1])
        return array_q

    @staticmethod
    def getInstance():
        if Quantizer.__instance is None:
            Quantizer()
        return Quantizer.__instance

def memcpy(dst: np.array, src: np.array, start: int = 0, count: int = -1) -> None:
    '''
    dst: to destination for src to copy to
    src: to data to copy from
    start: starting position in dst
    count: the number of elements to copy from src
    '''
    assert len(dst.shape) == 1 and len(src.shape) == 1
    if count == -1: count = src.size
    assert dst.size >= src.size+start
    dst[start:start+count] = src[:count]

if __name__ == '__main__':
    def test_quantizer():
        import skimage.io as io
        import skimage.transform as trans
        img = io.imread('./UNet/testData/0.png', as_gray=True)
        img = img / 255
        img = trans.resize(image=img, output_shape=(256, 256, 1))
        Quantizer.getInstance().setWordLength(8)
        print (Quantizer.getInstance().quantize(img[128:158, 128:158]))

    def test_memcpy():
        dst = np.zeros(shape=(20,))
        src = np.ones(shape=(10,))
        print ('before', dst)
        memcpy(dst, src, start=1, count=5)
        print ('after', dst)

    # test_memcpy()
    # test_quantizer()
