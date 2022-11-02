from tensorflow import tf

class LabelCleaner(tf.keras.Model):
    """
    This model is used to learn a mapping from the
    noisy labels to the clean labels from the validated
    dataset.

    The model is trained using an L1-loss function:
    `L = sum_i |y_i - y_hat_i|`

    where `y_i` is the clean label and `y_hat_i` is the
    predicted clean label.

    Parameters:
    -----------
    CNN: tf.keras.Model
        The base CNN model used to extract features from the
        images.
    """

    def __init__(self, CNN: tf.keras.Model):
        super(LabelCleaner, self).__init__()

        # Base CNN model
        self.CNN = CNN
        
        # Fully connected dense layers
        self.fc_1 = tf.keras.layers.Dense(units = 20, use_bias=False)
        self.fc_2 = tf.keras.layers.Dense(units = 256)
        self.fc_3 = tf.keras.layers.Dense(units = 256, use_bias=False, activation = "relu")
        self.fc_4 = tf.keras.layers.Dense(units = 10, use_bias=False,)
        
        # Batch Normalization layers
        self.bn_1 = tf.keras.layers.BatchNormalization()
        self.bn_2 = tf.keras.layers.BatchNormalization()
        self.bn_3 = tf.keras.layers.BatchNormalization()

    def call(self, inputs):
        img, y = inputs

        # Get the CNN output
        x = self.CNN(img)

        # Embed the output of the CNN to the noisy labels
        x = tf.concat([x, y], axis = 1)
        
        x = self.fc_1(x)    # Linear followed by batch normalization
        x = self.bn_1(x)

        x = self.fc_2(x)    # Linear followed by batch normalization
        x = self.bn_2(x)

        x = self.fc_3(x)    # ReLU

        x = self.fc_4(x)    # Linear followed by batch normalization
        x = self.bn_3(x)

        x = x + y           # Residual connection
        x = tf.clip_by_value(x, 0, 1)

        return x


class ImageClassifier(tf.keras.Model):
    """
    This model is used to classify the images. 
    It uses the output from the LabelCleaner model
    as input.

    Parameters:
    -----------
    CNN: tf.keras.Model
        The base CNN model used to extract features from the
        images.
    """

    def __init__(self, cnn: tf.keras.Model):
        super(ImageClassifier, self).__init__()

        self.cnn = cnn
        self.fc_1 = tf.keras.layers.Dense(units = 512),
        self.fc_1 = tf.keras.layers.Dense(units = 10, activation = "sigmoid")

    def call(self, inputs):
        x = self.cnn(inputs)
        x = self.fc_1(x)
        x = self.fc_2(x)

        return x