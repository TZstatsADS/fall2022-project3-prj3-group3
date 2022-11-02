import tensorflow as tf

@tf.function
def l1_loss(y_true, y_pred):
    return tf.reduce_sum(tf.abs(y_true - y_pred))

def get_custom_cross_entropy(n_verified, cleaned_labels):
    pi = tf.Variable(tf.ones((n_verified)), dtype = tf.float32)
    pj = tf.Variable(tf.ones((len(cleaned_labels))), dtype = tf.float32)
    c = tf.constant(cleaned_labels)

    def cross_entropy(y_true, y_pred):
        v = y_true[:n_verified]
        pj = y_pred[:n_verified]
        
        c = y_true[n_verified:]
        pi = y_pred[n_verified:]
        
        a = tf.keras.losses.categorical_crossentropy(v, pi)
        b = tf.keras.losses.categorical_crossentropy(c, pj)

        return a + b
    return tf.function(cross_entropy)
