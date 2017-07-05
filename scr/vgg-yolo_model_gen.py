from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, InputLayer, Dropout
from keras.layers.normalization import BatchNormalization

# 入力サイズ等はここを変更
width = 416
height = 416
r_w = 13
r_h = 13
r_n = 5
classes = 20


def vgg_yolo_model():
    model = Sequential()
    model.add(InputLayer(input_shape=(width, height, 3)))

    model.add(Conv2D(64, activation='relu',
                     padding='same', kernel_size=(3, 3), strides=(1, 1)))
    model.add(Conv2D(64, activation='relu',
                     padding='same', kernel_size=(3, 3), strides=(1, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same', strides=(2, 2)))

    model.add(Conv2D(128, data_format="channels_last",  activation='relu',
                     padding='same', kernel_size=(3, 3), strides=(1, 1)))
    model.add(Conv2D(128, data_format="channels_last",  activation='relu',
                     padding='same', kernel_size=(3, 3), strides=(1, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same', strides=(2, 2)))

    model.add(Conv2D(256, data_format="channels_last",  activation='relu',
                     padding='same', kernel_size=(3, 3), strides=(1, 1)))
    model.add(Conv2D(256, data_format="channels_last",  activation='relu',
                     padding='same', kernel_size=(3, 3), strides=(1, 1)))
    model.add(Conv2D(256, data_format="channels_last",  activation='relu',
                     padding='same', kernel_size=(3, 3), strides=(1, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same', strides=(2, 2)))

    model.add(Conv2D(512, data_format="channels_last",  activation='relu',
                     padding='same', kernel_size=(3, 3), strides=(1, 1)))
    model.add(Conv2D(512, data_format="channels_last",  activation='relu',
                     padding='same', kernel_size=(3, 3), strides=(1, 1)))
    model.add(Conv2D(512, data_format="channels_last",  activation='relu',
                     padding='same', kernel_size=(3, 3), strides=(1, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same', strides=(2, 2)))

    model.add(Conv2D(512, data_format="channels_last",  activation='relu',
                     padding='same', kernel_size=(3, 3), strides=(1, 1)))
    model.add(Conv2D(512, data_format="channels_last",  activation='relu',
                     padding='same', kernel_size=(3, 3), strides=(1, 1)))
    model.add(Conv2D(512, data_format="channels_last",  activation='relu',
                     padding='same', kernel_size=(3, 3), strides=(1, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same', strides=(2, 2)))

    #natu
    model.add(BatchNormalization())

    model.add(Dropout(0.25))
    model.add(Conv2D(125, use_bias=True, data_format="channels_last", #activation='liner',
                     padding='same', kernel_size=(1, 1), strides=(1, 1), kernel_initializer='random_uniform'))

    return model


def transfer_weights(my_model):
    from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
    def_model = VGG16(include_top=True, weights='imagenet', input_tensor=None, input_shape=None)

    # レイヤー単位で、17層までコピー。18層は全結合層なので不要。
    for i, l in enumerate(def_model.layers):
        if len(l.weights) == 0:
            continue

        # layer[17]までコピー
        if i >= 18:
            break

        w = l.get_weights()
        my_model.layers[i].set_weights(w)

        print('set weights layer[%d] %s' % (i, l.name))


# モデルの構築と表示
vgg_yolo_model = vgg_yolo_model()
vgg_yolo_model.summary()

# データを移す
transfer_weights(vgg_yolo_model)
# 名前をつけて保存する。
vgg_yolo_model.save_weights('weight/vgg_yolo_def.h5')

with open('output/vgg-yolo.json', 'w') as fp:
    json_string = vgg_yolo_model.to_json()
    fp.write(json_string)
