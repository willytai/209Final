from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler

# a little different from the original model
# notice the zero padding and the dimensions are different
# actually makes it easier for us to do concatenation
def unet(pretrained_weights = None,input_size = (256,256,1)):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    model = Model(inputs = inputs, outputs = conv10)

    model.compile(optimizer = Adam(learning_rate = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])

    model.summary()

    if(pretrained_weights):
        model.load_weights(pretrained_weights)

    return model


if __name__ == '__main__':
    model = unet()
    exit()
    model.load_weights("unet_model.hdf5")
    layer1 = model.get_layer('conv2d')
    layer2 = model.get_layer('conv2d_1')
    layer3 = model.get_layer('max_pooling2d')
    layer4 = model.get_layer('conv2d_2')
    layer5 = model.get_layer('conv2d_3')
    layer6 = model.get_layer('max_pooling2d_1')
    layer7 = model.get_layer('conv2d_4')
    layer8 = model.get_layer('conv2d_5')
    layer9 = model.get_layer('max_pooling2d_2')
    layer10 = model.get_layer('conv2d_6')
    # _1_layer_model = Model(inputs = model.input, outputs = layer1.output)
    # _2_layer_model = Model(inputs = model.input, outputs = layer2.output)
    # _3_layer_model = Model(inputs = model.input, outputs = layer3.output)
    # _4_layer_model = Model(inputs = model.input, outputs = layer4.output)
    # _5_layer_model = Model(inputs = model.input, outputs = layer5.output)
    # _6_layer_model = Model(inputs = model.input, outputs = layer6.output)

    _7_layer_model = Model(inputs = model.input, outputs = layer7.output)
    _8_layer_model = Model(inputs = model.input, outputs = layer8.output)
    _9_layer_model = Model(inputs = model.input, outputs = layer9.output)
    # _10_layer_model = Model(inputs = model.input, outputs = layer10.output)

    # weights = layer1.get_weights()[0]
    # bias = layer1.get_weights()[1]
    # print ('weights', weights[:,:,:,0])
    # print ('bias', bias)

    import skimage.io as io
    import skimage.transform as trans
    import numpy as np
    img = io.imread('./testData/0.png', as_gray=True)
    img = img / 255
    img = trans.resize(image=img, output_shape=(256, 256, 1))
    # _1_out = _1_layer_model.predict(np.ones((1, 256, 256, 1)))[0]
    # _1_out = _1_layer_model.predict(img.reshape((1, 256, 256, 1)))[0]
    # _2_out = _2_layer_model.predict(img.reshape((1, 256, 256, 1)))[0]
    # _3_out = _3_layer_model.predict(img.reshape((1, 256, 256, 1)))[0]
    # _4_out = _4_layer_model.predict(img.reshape((1, 256, 256, 1)))[0]
    # _5_out = _5_layer_model.predict(img.reshape((1, 256, 256, 1)))[0]
    # _6_out = _6_layer_model.predict(img.reshape((1, 256, 256, 1)))[0]

    _7_out = _7_layer_model.predict(img.reshape((1, 256, 256, 1)))[0]
    _8_out = _8_layer_model.predict(img.reshape((1, 256, 256, 1)))[0]
    _9_out = _9_layer_model.predict(img.reshape((1, 256, 256, 1)))[0]
    # _10_out = _10_layer_model.predict(img.reshape((1, 256, 256, 1)))[0]

    # np.save('layer1_conv_output_golden.npy', _1_out)
    # np.save('layer2_conv_output_golden.npy', _2_out)
    # np.save('layer1_maxpool_output_golden.npy', _3_out)
    # np.save('layer3_conv_output_golden.npy', _4_out)
    # np.save('layer4_conv_output_golden.npy', _5_out)
    # np.save('layer2_maxpool_output_golden.npy', _6_out)

    np.save('layer5_conv_output_golden.npy', _7_out)
    np.save('layer6_conv_output_golden.npy', _8_out)
    np.save('layer3_maxpool_output_golden.npy', _9_out)
    # np.save('layer7_conv_output_golden.npy', _10_out)

    # with open('unet_model.json', 'w') as f:
    #     f.write(model.to_json())
