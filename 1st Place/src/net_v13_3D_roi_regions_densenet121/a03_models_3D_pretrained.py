# coding: utf-8
__author__ = 'ZFTurbo: https://www.drivendata.org/users/ZFTurbo/'


if __name__ == '__main__':
    import os

    gpu_use = 0
    print('GPU use: {}'.format(gpu_use))
    os.environ["KERAS_BACKEND"] = "tensorflow"
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_use)


from a00_common_functions import *
from tf_keras.models import Model
from tf_keras.layers import BatchNormalization, SpatialDropout3D, Activation, Concatenate, Input, Conv3D, MaxPooling3D, UpSampling3D
from tf_keras import backend as K


def Model_3D_pretrained_densenet121(
        input_shape,
        dropout_val=0.2,
        out_channels=1,
        use_imagenet=True,
):
    from tf_keras.models import Model, load_model
    from tf_keras.layers import Dense, Input, GlobalAveragePooling3D, Dropout
    from classification_models_3D.keras import Classifiers

    type = 'densenet121'
    backbone, preproc_input = Classifiers.get(type)
    model_upd = backbone(include_top=False,
                         weights=None,
                         input_shape=input_shape,
                         pooling='avg', )
    # https://datascience.stackexchange.com/questions/56087/why-do-people-import-weights-for-densenet-when-keras-includes-them 
    # we can just try with weights=imagenet instead of downloading all the weights
    #if use_imagenet:
    #    model_upd.load_weights(MODELS_PATH + 'converter/{}_inp_channel_3.h5'.format(type))
    x = model_upd.layers[-1].output
    # x = GlobalAveragePooling3D()(x)
    x = Dense(512, activation='sigmoid', name='classification')(x)
    x = Dropout(dropout_val)(x)
    x = Dense(out_channels, activation='sigmoid', name='prediction')(x)
    model = Model(inputs=model_upd.inputs, outputs=x)
    # print(model.summary())
    return model, preproc_input


if __name__ == '__main__':
    # model, _ = Model_3D_pretrained_v2(input_shape=(96, 128, 160, 3))
    model, _ = Model_3D_pretrained_densenet121(input_shape=(96, 128, 128, 3))
    print(model.summary())
    print(get_model_memory_usage(1, model))