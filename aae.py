#coding:utf8
import keras.backend as K

from keras.optimizers import Adam
from keras.layers import Input, Dense, Lambda, BatchNormalization, Activation
from keras.models import Model
from keras.callbacks import EarlyStopping, History, ModelCheckpoint, CSVLogger

from utils.configuration import Config
from utils.distributions import log_Bernoulli
import numpy as np


class AdversarialAutoencoder(object):
  def __init__(self, config):
    if not isinstance(config, Config):
      raise TypeError("config 类型错误")

    self.discriminator_loss = 0.

    #配置
    self.config = config
    #创建模型
    self.vae, self.encoder, self.decoder, self.discriminator,self.vae_callbacks, self.discriminator_callbacks = self.__create_network()

  def __create_network(self):
    x = Input(shape = (self.config.original_dim,))
    pred_p_z= Input(shape = (1,))
    pred_q_z = Input(shape = (1,))
  
    #encoder
    #h1
    encoder_h1_dense = Dense(1000, name = "encoder_h1")(x)
    encoder_h1_batch = BatchNormalization()(encoder_h1_dense)
    encoder_h1_activation = Activation(activation = "relu")(encoder_h1_batch)
    
    #h2
    encoder_h2_dense = Dense(1000, name = "encoder_h2")(encoder_h1_activation)
    encoder_h2_batch = BatchNormalization()(encoder_h2_dense)
    encoder_h2_activation = Activation(activation = "relu")(encoder_h2_batch)
  
    #h3
    encoder_h3_dense = Dense(self.config.latent_dim, name = "encoder_h3")(encoder_h2_activation)
    hidden_z =encoder_h3_batch = BatchNormalization()(encoder_h3_dense)
  
    #encoder 模型
    encoder = Model(x, hidden_z)
    from keras.utils import plot_model
    plot_model(encoder, to_file='encoder.png', show_shapes=True)
  
    #decoder
    #h1
    layer_decoder_h1_dense = Dense(1000, name = "decoder_h1")
    decoder_h1_dense = layer_decoder_h1_dense(hidden_z)
    layer_decoder_batch1 = BatchNormalization()
    decoder_h1_batch = layer_decoder_batch1(decoder_h1_dense)
    decoder_h1_activation = Activation(activation = "relu")(decoder_h1_batch)
    #h2
    layer_decoder_h2_dense = Dense(1000, name = "decoder_h2")
    decoder_h2_dense = layer_decoder_h2_dense(decoder_h1_activation)
    layer_decoder_batch2 = BatchNormalization()
    decoder_h2_batch = layer_decoder_batch2(decoder_h2_dense)
    decoder_h2_activation = Activation(activation = "relu")(decoder_h2_batch)
    #h3
    layer_decoder_h3_dense = Dense(self.config.original_dim, activation = "sigmoid", name = "decoder_h3")
    generate_x = decoder_h3_dense = layer_decoder_h3_dense(decoder_h2_activation)
  
    #vae 模型
    vae = Model([pred_p_z, pred_q_z, x], generate_x)
    plot_model(vae, to_file = "vae.png", show_shapes = True)
  
    #Discriminator
    input_discriminator = Input(shape = (self.config.latent_dim,))
    #input_discriminator_lambda = Lambda(lambda x: x, name = "discriminator_lambda_input")(input_discriminator)
    discriminator_h1_dense = Dense(500, activation = "relu", name = "discriminator_h1")(input_discriminator)
    discriminator_h2_dense = Dense(500, activation = "relu", name = "discriminator_h2")(discriminator_h1_dense)
    discriminator_h3_dense = Dense(1, activation = "sigmoid", name = "discriminator_h3")(discriminator_h2_dense)
    #discriminator 模型
    discriminator = Model(input_discriminator, discriminator_h3_dense)
    plot_model(discriminator, to_file = "discriminator.png", show_shapes = True)

    #decoder 模型，用于计算流图
    input_decoder = Input(shape = (self.config.latent_dim,))
    _decoder_h1_dense = layer_decoder_h1_dense(input_decoder)
    _decoder_h1_batch = layer_decoder_batch1(_decoder_h1_dense)
    _decoder_h1_activation = Activation(activation = "relu")(_decoder_h1_batch)
    _decoder_h2_dense = layer_decoder_h2_dense(_decoder_h1_activation)
    _decoder_h2_batch = layer_decoder_batch2(_decoder_h2_dense)
    _decoder_h2_activation = Activation(activation = "relu")(_decoder_h2_batch)
    _generate_x = _decoder_h3_dense = layer_decoder_h3_dense(_decoder_h2_activation)
    decoder = Model(input_decoder, _generate_x)
    plot_model(decoder, to_file = "decoder.png", show_shapes = True)
    
    #计算重构误差
    def create_reconstruction_loss(x, decode_x):
      data_type = self.config.data_type
      if data_type == "gray":
        return K.mean(K.sum((x - decode_x)**2, axis=1), axis = 0)
      elif data_type == "binary":
        return -K.mean(K.sum(x * K.log(decode_x) + (1. - x) * K.log(1. - decode_x), axis=1), axis = 0)

    def discriminator_loss_func(y, pred_y):
      return  2. * -K.mean(K.sum(log_Bernoulli(y, pred_y), axis = 1), axis = 0)

    """
    创建误差函数
    """
    def vae_loss_func(noise1, noise2):
      def vae_loss(x, decode_x):
        #重构误差
        reconstruction_loss = create_reconstruction_loss(x, decode_x)

        #判别器误差
       # z_pred = self.discriminator_z_pred #__sample_from_prior()
       # z = self.discriminator_z
        pred_p_z = noise1
        pred_q_z = noise2
        discriminator_loss = -K.mean(K.log(pred_p_z) + (K.log(1.-pred_q_z)))
        return reconstruction_loss - self.config.lamb * discriminator_loss
      return vae_loss
    def metrics_vae_discriminator_loss(pred_p_z, pred_q_z):
      def discriminator_loss(x, decode_x):
        return -K.mean(K.log(pred_p_z) + K.log(1. - pred_q_z))
      return discriminator_loss

    #创建metrics显示误差
    def metrics_reconstruction(x, decode_x):
      return create_reconstruction_loss(x, decode_x)


    #判别器准确率
    def metrics_discriminator_accuracy(y, pred_y):
      #return binary_accuracy(y, pred_y)
      binary_pred_y = K.cast(K.greater_equal(pred_y, 0.5), K.floatx())
      return K.mean(K.cast(K.equal(y, binary_pred_y), K.floatx()))

    #compile
    discriminator.compile(optimizer = Adam(lr = self.config.learning_rate),
                      loss = discriminator_loss_func, 
                      metrics = [metrics_discriminator_accuracy])

    vae.compile(optimizer = Adam(lr = self.config.learning_rate),
                      loss=vae_loss_func(pred_p_z, pred_q_z),
                      metrics = [metrics_reconstruction, metrics_vae_discriminator_loss(pred_p_z, pred_q_z)])

    # TRAINING
    # Callbacks
    vae_callbacks = []
    discriminator_callbacks = []

    history = History()
    vae_callbacks.append(history)

    csvlogger = CSVLogger(filename = str(self.config.latent_dim) + 'ave_training.log')
    vae_callbacks.append(csvlogger)

    checkpointer = ModelCheckpoint( filepath = str(self.config.latent_dim) + 'ave_weights.hdf5', verbose=0, save_best_only=True, save_weights_only=True) 
    vae_callbacks.append(checkpointer)

    history1 = History()
    discriminator_callbacks.append(history1)
    csvlogger1 = CSVLogger(filename= str(self.config.latent_dim) + 'discri_training.log')
    discriminator_callbacks.append(csvlogger1)

    checkpointer1 = ModelCheckpoint( filepath = str(self.config.latent_dim) + 'discri_weights.hdf5', verbose=0, save_best_only=True, save_weights_only=True )
    discriminator_callbacks.append(checkpointer1)

    return vae, encoder, decoder, discriminator, vae_callbacks, discriminator_callbacks

#def get_normalized_vector(v):
#    v = v / (1e-20 + T.max(T.abs_(v), axis=1, keepdims=True))
#    v_2 = T.sum(v**2,axis=1,keepdims=True)
#    return v / T.sqrt(1e-6+v_2)
#    ###### uniform ########
#    elif(z_prior is 'uniform'):
#        v = get_normalized_vector(self.rng.normal(size=z.shape,dtype=theano.config.floatX))
#        r = T.power(self.rng.uniform(size=z.sum(axis=1,keepdims=True).shape,low=0,high=1.0,dtype=theano.config.floatX),1./z.shape[1])
#        r = T.patternbroadcast(r,[False,True])
#        return 2.0*r*v
#  
#    else:
#        raise NotImplementedError()
  

