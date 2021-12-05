from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler

# a little different from the original model
# notice the zero padding and the dimensions are different
# actually makes it easier for us to do concatenation
def unet(pretrained_weights = None,input_size = (256,256,1)):
    inputs = Input(input_size)
    conv1 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    # drop4 = Dropout(0.5)(conv4)
    # pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
    # pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    # conv5 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    # conv5 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    # drop5 = Dropout(0.5)(conv5)

    # up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    # up6 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv5))
    # merge6 = concatenate([drop4,up6], axis = 3)
    # merge6 = concatenate([conv4,up6], axis = 3)
    # conv6 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    # conv6 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    # up7 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    up7 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv4))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(32, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(16, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
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
    model.load_weights("unet_model_light.hdf5")

    import skimage.io as io
    import skimage.transform as trans
    import numpy as np
    img = io.imread('./testData/0.png', as_gray=True)
    img = img / 255
    img = trans.resize(image=img, output_shape=(256, 256, 1)).reshape((1, 256, 256, 1))

    layer = model.get_layer('conv2d_8')
    test = Model(inputs=layer.input, outputs=layer.output)
    inn = np.zeros((1,64,64,128))
    inn[0, :, :, 0] = np.ones((64, 64))
    check = test.predict(inn)[0]
    print ('kernel\n')
    print (layer.get_weights()[0][:,:,0,0])
    print ('bias\n')
    print (layer.get_weights()[1][0])
    print ('resut\n')
    print (check[-2:, -2:, 0])
    exit()

    conv_count = 1
    pool_count = 1
    up_sample_count = 1
    concat_count = 1

    for layer in model.layers:
        print (layer.name)
        if layer.__class__.__name__ == 'InputLayer':
            continue
        elif layer.__class__.__name__ == 'Conv2D':
            test = Model(inputs=model.input, outputs=layer.output)
            ret = test.predict(img)[0]
            np.save('../intermediate_feature_maps_light_golden/layer{}_conv_output_golden.npy'.format(conv_count), ret)
            conv_count += 1
        elif layer.__class__.__name__ == 'MaxPooling2D':
            test = Model(inputs=model.input, outputs=layer.output)
            ret = test.predict(img)[0]
            np.save('../intermediate_feature_maps_light_golden/layer{}_maxpool_output_golden.npy'.format(pool_count), ret)
            pool_count += 1
        elif layer.__class__.__name__ == 'UpSampling2D':
            test = Model(inputs=model.input, outputs=layer.output)
            ret = test.predict(img)[0]
            np.save('../intermediate_feature_maps_light_golden/layer{}_upsample_output_golden.npy'.format(up_sample_count), ret)
            up_sample_count += 1
        elif layer.__class__.__name__ == 'Concatenate':
            test = Model(inputs=model.input, outputs=layer.output)
            ret = test.predict(img)[0]
            np.save('../intermediate_feature_maps_light_golden/layer{}_concatenate_output_golden.npy'.format(concat_count), ret)
            concat_count += 1
        else: break
