import tensorflow as tf
import tensorflow.contrib.layers as layers
from keras.layers import LSTM
from keras.layers import Reshape
def _cnn_rnn_mlp(convs,hiddens,dueling,inpt,num_actions,scope,reuse=False,layer_norm=False):
    with tf.variable_scope(scope, reuse=reuse):
        out = inpt
        with tf.variable_scope("convnet"):
            for num_outputs, kernel_size, stride in convs:
                out = layers.convolution2d(out,
                                           num_outputs=num_outputs,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           activation_fn=tf.nn.relu)
        conv_out = layers.flatten(out)
        with tf.variable_scope("action_value"):
            action_out=LSTM(hiddens[0],activation='relu')(action_out)
            for hidden in hiddens[1:]:
                action_out = layers.fully_connected(action_out, num_outputs=hidden, activation_fn=None)
                if layer_norm:
                    action_out = layers.layer_norm(action_out, center=True, scale=True)
                action_out = tf.nn.relu(action_out)
            action_scores = layers.fully_connected(action_out, num_outputs=num_actions, activation_fn=None)

        if dueling:
            with tf.variable_scope("state_value"):
                state_out = conv_out
                state_out = LSTM(hiddens[0], activation='relu')(state_out)
                for hidden in hiddens[1:]:
                    state_out = layers.fully_connected(state_out, num_outputs=hidden, activation_fn=None)
                    if layer_norm:
                        state_out = layers.layer_norm(state_out, center=True, scale=True)
                    state_out = tf.nn.relu(state_out)
                state_score = layers.fully_connected(state_out, num_outputs=1, activation_fn=None)
            action_scores_mean = tf.reduce_mean(action_scores, 1)
            action_scores_centered = action_scores - tf.expand_dims(action_scores_mean, 1)
            q_out = state_score + action_scores_centered
        else:
            q_out = action_scores
        return q_out
def cnn_rnn_mlp(convs,hiddens,dueling=False,layer_norm=False):
    return lambda *args, **kwargs: _cnn_rnn_mlp(convs, hiddens,dueling, layer_norm=layer_norm, *args, **kwargs)


def VIN(X, vin_config):
    """
    Value Iteration Network
    X : (?, m, n, ch_i) - batch of images (stack of gridworld and goal)
        - gridworld (m, n) = grid with 1 and 0 ;
        - goal (m, n) = grid with 10 at goal position
    vin_config: the VIN config (type VINConfig)
        - k: Number of Value Iteration computations
        - ch_h : Channels in initial hidden layer
        - ch_q : Channels in q layer (~actions)
    """

    h = conv2d(inputs=X, filters=vin_config.ch_h, name='h0', use_bias=True)
    r = conv2d(inputs=h, filters=1, name='r')

    # Initialize value map (zero everywhere)
    v = tf.zeros_like(r)

    rv = tf.concat([r, v], axis=3)
    q = conv2d(inputs=rv, filters=vin_config.ch_q, name='q', reuse=None)  # Initial set before sharing weights
    v = tf.reduce_max(q, axis=3, keepdims=True, name='v')

    # K iterations of VI module
    for _ in range(vin_config.k):
        rv = tf.concat([r, v], axis=3)
        q = conv2d(inputs=rv, filters=vin_config.ch_q, name='q', reuse=True)  # Sharing weights
        v = tf.reduce_max(q, axis=3, keepdims=True, name='v')

    rv = tf.concat([r, v], axis=3)
    return rv
