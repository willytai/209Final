import numpy as np

def quantize8(value: np.array, fl: int) -> np.array:
    wl = 8
    '''
    wl: word length
    fl: fraction length

    For quantizing normalized images, it's basically quantize(image, 8, 7)
    since pixel value ranges between [0, 1]
    '''
    intLen = wl - fl
    precision = pow(2, -fl)
    valueRange = (-pow(2, intLen-1) + precision, pow(2, intLen-1) - precision) # cutoff values

    # some tricks (float -> fixed -> float)
    value_q = value * pow(2, fl)
    value_q = np.round(value_q)
    value_q = value_q * pow(2, -fl)

    # saturate the values
    value_q = np.maximum(value_q, valueRange[0])
    value_q = np.minimum(value_q, valueRange[1])

    return value_q

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
    def test_quantize8():
        import skimage.io as io
        import skimage.transform as trans
        img = io.imread('./UNet/testData/0.png', as_gray=True)
        img = img / 255
        img = trans.resize(image=img, output_shape=(256, 256, 1))
        print (quantize(img[128:158, 128:158], 8, 7))

    def test_memcpy():
        dst = np.zeros(shape=(20,))
        src = np.ones(shape=(10,))
        print ('before', dst)
        memcpy(dst, src, start=1, count=5)
        print ('after', dst)

    # test_memcpy()
    # test_quantize8()
