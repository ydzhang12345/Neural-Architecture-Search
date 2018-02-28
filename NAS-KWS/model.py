from keras.models import Model
from keras.layers import Input, Dense, Conv2D, GlobalAveragePooling2D, BatchNormalization, SeparableConv2D, Activation, Flatten
import pdb

def model_fn_baack(fingerprint_input, model_settings, model_size_info, 
                          is_training):
  """Builds a model with depthwise separable convolutional neural network
  Model definition is based on https://arxiv.org/abs/1704.04861 and
  Tensorflow implementation: https://github.com/Zehaos/MobileNet

  model_size_info: defines number of layers, followed by the DS-Conv layer
    parameters in the order {number of conv features, conv filter height, 
    width and stride in y,x dir.} for each of the layers. 
  Note that first layer is always regular convolution, but the remaining 
    layers are all depthwise separable convolutions.
  """

  def ds_cnn_arg_scope(weight_decay=0):
    """Defines the default ds_cnn argument scope.
    Args:
      weight_decay: The weight decay to use for regularizing the model.
    Returns:
      An `arg_scope` to use for the DS-CNN model.
    """
    with slim.arg_scope(
        [slim.convolution2d, slim.separable_convolution2d],
        weights_initializer=slim.initializers.xavier_initializer(),
        biases_initializer=slim.init_ops.zeros_initializer(),
        weights_regularizer=slim.l2_regularizer(weight_decay)) as sc:
      return sc

  def _depthwise_separable_conv(inputs,
                                num_pwc_filters,
                                sc,
                                kernel_size,
                                stride):
    """ Helper function to build the depth-wise separable convolution layer.
    """

    # skip pointwise by setting num_outputs=None
    depthwise_conv = slim.separable_convolution2d(inputs,
                                                  num_outputs=None,
                                                  stride=stride,
                                                  depth_multiplier=1,
                                                  kernel_size=kernel_size,
                                                  scope=sc+'/depthwise_conv')

    bn = slim.batch_norm(depthwise_conv, scope=sc+'/dw_batch_norm')
    pointwise_conv = slim.convolution2d(bn,
                                        num_pwc_filters,
                                        kernel_size=[1, 1],
                                        scope=sc+'/pointwise_conv')
    bn = slim.batch_norm(pointwise_conv, scope=sc+'/pw_batch_norm')
    return bn


  if is_training:
    dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')

  label_count = model_settings['label_count']
  input_frequency_size = model_settings['dct_coefficient_count']
  input_time_size = model_settings['spectrogram_length']
  fingerprint_4d = tf.reshape(fingerprint_input,
                              [-1, input_time_size, input_frequency_size, 1])
 
  t_dim = input_time_size
  f_dim = input_frequency_size

  #Extract model dimensions from model_size_info
  num_layers = model_size_info[0]
  conv_feat = [None]*num_layers
  conv_kt = [None]*num_layers
  conv_kf = [None]*num_layers
  conv_st = [None]*num_layers
  conv_sf = [None]*num_layers
  i=1
  for layer_no in range(0,num_layers):
    conv_feat[layer_no] = model_size_info[i]
    i += 1
    conv_kt[layer_no] = model_size_info[i]
    i += 1
    conv_kf[layer_no] = model_size_info[i]
    i += 1
    conv_st[layer_no] = model_size_info[i]
    i += 1
    conv_sf[layer_no] = model_size_info[i]
    i += 1

  scope = 'DS-CNN'
  with tf.variable_scope(scope) as sc:
    end_points_collection = sc.name + '_end_points'
    with slim.arg_scope([slim.convolution2d, slim.separable_convolution2d],
                        activation_fn=None,
                        weights_initializer=slim.initializers.xavier_initializer(),
                        biases_initializer=slim.init_ops.zeros_initializer(),
                        outputs_collections=[end_points_collection]):
      with slim.arg_scope([slim.batch_norm],
                          is_training=is_training,
                          decay=0.96,
                          updates_collections=None,
                          activation_fn=tf.nn.relu):
        for layer_no in range(0,num_layers):
          if layer_no==0:
            net = slim.convolution2d(fingerprint_4d, conv_feat[layer_no],\
                      [conv_kt[layer_no], conv_kf[layer_no]], stride=[conv_st[layer_no], conv_sf[layer_no]], padding='SAME', scope='conv_1')
            net = slim.batch_norm(net, scope='conv_1/batch_norm')
          else:
            net = _depthwise_separable_conv(net, conv_feat[layer_no], \
                      kernel_size = [conv_kt[layer_no],conv_kf[layer_no]], \
                      stride = [conv_st[layer_no],conv_sf[layer_no]], sc='conv_ds_'+str(layer_no))
          t_dim = math.ceil(t_dim/float(conv_st[layer_no]))
          f_dim = math.ceil(f_dim/float(conv_sf[layer_no]))

        net = slim.avg_pool2d(net, [t_dim, f_dim], scope='avg_pool')

    net = tf.squeeze(net, [1, 2], name='SpatialSqueeze')
    logits = slim.fully_connected(net, label_count, activation_fn=None, scope='fc1')

  if is_training:
    return logits, dropout_prob
  else:
    return logits




# generic model design
def model_fn(actions):
    # unpack the actions from the list
    filters_1, kernel_1, stride_1, filters_2, kernel_2, stride_2, filters_3, kernel_3, stride_3 = actions

    ip = Input(shape=(49, 10, 1))
    y = Conv2D(64, (10, 4), strides=(2, 2), padding='same', activation='relu')(ip)
    y = BatchNormalization()(y)
    y = SeparableConv2D(filters_1, (kernel_1, kernel_1), strides=(stride_1, stride_1), padding='same', activation='relu')(y)
    y = BatchNormalization()(y)
    y = SeparableConv2D(filters_2, (kernel_2, kernel_2), strides=(stride_2, stride_2), padding='same', activation='relu')(y)
    y = BatchNormalization()(y)
    y = SeparableConv2D(filters_3, (kernel_3, kernel_3), strides=(stride_3, stride_3), padding='same', activation='relu')(y)
    y = BatchNormalization()(y)
    y = SeparableConv2D(filters_3, (kernel_3, kernel_3), strides=(stride_3, stride_3), padding='same', activation='relu')(y)
    y = BatchNormalization()(y)
    y = GlobalAveragePooling2D()(y)
    
    y = Dense(12, activation='softmax')(y)

    model = Model(ip, y)
    return model