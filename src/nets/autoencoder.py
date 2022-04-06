from tensorflow.keras.models import Model
from tensorflow.keras import Sequential, layers
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np


class AutoEncoder(Model):
    def __init__(self, input_shape, filters_per_layer, latent_dim):
        super(AutoEncoder, self).__init__()

        shapes = [input_shape]
        for num_filters in filters_per_layer:
            shapes.append((
                shapes[-1][0] // 2,
                shapes[-1][1] // 2,
                num_filters,
            ))

        encoder_layers = [layers.Input(shape=input_shape)]
        for i in range(1, len(shapes)):
            params = AutoEncoder.compute_conv_params_2d(shapes[i-1], shapes[i], 3, 2)

            padding = np.array(params["pad_crop"])
            padding = np.where(padding > 0, padding, 0)
            if np.sum(padding) > 0:
                encoder_layers.append(layers.ZeroPadding2D(padding=padding))

            encoder_layers.append(
                layers.Conv2D(
                    params["num_filters"],
                    kernel_size=params["kernel_size"],
                    strides=params["strides"],
                    activation="relu",
                )
            )

            cropping = np.array(params["pad_crop"])
            cropping = np.where(cropping < 0, cropping, 0)
            if np.sum(cropping) < 0:
                encoder_layers.append(layers.Cropping2D(cropping=cropping))

        encoder_layers.append(layers.Flatten())
        encoder_layers.append(layers.Dense(latent_dim, activation="relu"))

        decoder_layers = [
            layers.Dense(np.product(shapes[-1]), activation="relu"),
            layers.Reshape(shapes[-1]),
        ]
        for i in range(len(shapes) - 2, -1, -1):
            params = AutoEncoder.compute_conv_trans_params_2d(shapes[i+1], shapes[i], 3, 2)

            padding = np.array(params["pad_crop"])
            padding = np.where(padding > 0, padding, 0)
            if np.sum(padding) > 0:
                decoder_layers.append(layers.ZeroPadding2D(padding=padding))

            decoder_layers.append(
                layers.Conv2DTranspose(
                    params["num_filters"],
                    kernel_size=params["kernel_size"],
                    strides=params["strides"],
                    activation="relu",
                )
            )

            cropping = np.array(params["pad_crop"])
            cropping = np.where(cropping < 0, -cropping, 0)
            if np.sum(cropping) > 0:
                decoder_layers.append(layers.Cropping2D(cropping=cropping))

        self.encoder = Sequential(encoder_layers)
        self.decoder = Sequential(decoder_layers)


    def call(self, x):
        return self.decoder(self.encoder(x))

    ##############################
    ### CONVOLUTIONAL ALGEBRA: ###
    ##############################

    @staticmethod
    def compute_conv_pad_crop_1d(input_len, output_len, kernel_size, strides):
        direct_len = AutoEncoder.get_conv_shape_1d(input_len, kernel_size, strides)

        # If the output has too much length, we need to crop it.
        if output_len < direct_len:
            cropping = direct_len - output_len
            if cropping % 2 == 0:
                return (-(cropping // 2), -(cropping // 2))
            else:
                return (-(cropping // 2), -(cropping // 2 + 1))

        # Otherwise we need to pad it.
        padding = strides * (output_len - direct_len)
        if padding % 2 == 0:
            return (padding // 2, padding // 2)
        else:
            return (padding // 2, padding // 2 + 1)


    @staticmethod
    def compute_conv_params_2d(input_shape, output_shape, kernel_size, strides):
        return {
            "strides": strides,
            "kernel_size": kernel_size,
            "num_filters": output_shape[2],
            "pad_crop": (
                AutoEncoder.compute_conv_pad_crop_1d(input_shape[0], output_shape[0], kernel_size, strides),
                AutoEncoder.compute_conv_pad_crop_1d(input_shape[1], output_shape[1], kernel_size, strides),
            ),
        }


    @staticmethod
    def get_conv_shape_1d(input_len, kernel_size, strides):
        return (input_len - kernel_size) // strides + 1


    @staticmethod
    def get_conv_shape_2d(input_shape, num_filters, kernel_size, strides):
        return (
            AutoEncoder.get_conv_shape_1d(input_shape[0], kernel_size, strides),
            AutoEncoder.get_conv_shape_1d(input_shape[1], kernel_size, strides),
            num_filters,
        )


    @staticmethod
    def compute_conv_trans_pad_crop_1d(input_len, output_len, kernel_size, strides):
        direct_len = AutoEncoder.get_conv_trans_shape_1d(input_len, kernel_size, strides)

        # If the output has too much length, we need to crop it.
        if output_len < direct_len:
            cropping = direct_len - output_len
            if cropping % 2 == 0:
                return (-(cropping // 2), -(cropping // 2))
            else:
                return (-(cropping // 2), -(cropping // 2 + 1))

        # Otherwise we need to pad it.
        padding = strides * (output_len - direct_len)
        if padding % 2 == 0:
            return (padding // 2, padding // 2)
        else:
            return (padding // 2, padding // 2 + 1)


    @staticmethod
    def compute_conv_trans_params_2d(input_shape, output_shape, kernel_size, strides):
        return {
            "strides": strides,
            "kernel_size": kernel_size,
            "num_filters": output_shape[2],
            "pad_crop": (
                AutoEncoder.compute_conv_trans_pad_crop_1d(input_shape[0], output_shape[0], kernel_size, strides),
                AutoEncoder.compute_conv_trans_pad_crop_1d(input_shape[1], output_shape[1], kernel_size, strides),
            ),
        }


    @staticmethod
    def get_conv_trans_shape_1d(input_len, kernel_size, strides):
        return kernel_size + (input_len - 1) * strides


    @staticmethod
    def get_conv_trans_shape_2d(input_shape, num_filters, kernel_size, strides):
        return (
            AutoEncoder.get_conv_trans_shape_1d(input_shape[0], kernel_size, strides),
            AutoEncoder.get_conv_trans_shape_1d(input_shape[1], kernel_size, strides),
            num_filters,
        )
