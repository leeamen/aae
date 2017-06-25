#coding:utf8

import matplotlib
import matplotlib.pyplot as plt
#%matplotlib inline

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import keras.backend as K
from aae import AdversarialAutoencoder
from utils.configuration import Config
import numpy as np
np.random.seed(np.random.randint(1234))

from utils.load_data import load_dataset
import sample_test as sample
import load_data


import tensorflow as tf
tf.set_random_seed(1234)

epsilon = 1e-7
def log_Bernoulli( sample, probs ):
    #逐元素clip（将超出指定范围的数强制变为边界值）
    probs =  np.clip(probs, epsilon, 1.-epsilon)
    return sample * np.log(probs) + (1. - sample) * np.log( 1. - probs )


"""
采样p N(0, 1)
"""
def sample_from_prior(z_q):
  z_prior = config.z_prior
  ###### gausssian #######
  if(z_prior is "gaussian"):
    return K.eval(K.random_normal(shape = z_q.shape, dtype = "float32"))
  elif z_prior is "uniform":
    return K.eval(K.random_uniform(shape = z_q.shape, dtype = "float32"))



def next_batch(ith, batch_size):
  x = 1.* x_train[ith*batch_size: (ith+1)*batch_size, :]
  return x, None

def train():
  training_epochs = config.number_epochs
  number_of_batches = int(np.shape(x_train)[0] / config.batch_size)
  print("total_batch:%d"%(number_of_batches))
  print("x_train shape:", x_train.shape)
  print("x_test shape:", x_test.shape)

  # Training cycle
  for epoch in range(training_epochs):
    avg_cost = 0.
    avg_recon_cost = 0.
    avg_discriminator_loss = 0.
    # Loop over all batches
    for i in range(number_of_batches):
      batch_x, _ = next_batch(i, config.batch_size)

      #encoder生成样本
      q_z = aae_model.encoder.predict(batch_x, batch_size = config.batch_size)

      label_q_z = np.array(np.array([[0.]]*config.batch_size, dtype = np.float32))
      #p_z 采样使用theano作者的采样方法
      p_z = sample.sample_from_prior(q_z, config.z_prior)
      label_p_z = np.array(np.array([[1.]]*config.batch_size, dtype = np.float32))
      z_train = np.vstack((q_z, p_z))
      z_train_y = np.vstack((label_q_z, label_p_z))

      #判别器
      aae_model.discriminator.fit(x = z_train, y = z_train_y, epochs = 1, batch_size = config.batch_size,
                                  validation_data = (z_train, z_train_y), callbacks = aae_model.discriminator_callbacks, verbose = 0)

      pred_q_z = aae_model.discriminator.predict(q_z, batch_size = config.batch_size)
      pred_p_z = aae_model.discriminator.predict(p_z, batch_size = config.batch_size)
      
      aae_model.vae.fit(x = [pred_p_z, pred_q_z, batch_x], y = batch_x, epochs = 1, 
                              validation_data = ([pred_p_z, pred_q_z, batch_x], batch_x), batch_size = config.batch_size, callbacks = aae_model.vae_callbacks, verbose = 0)
     
      # Compute average loss
      cost, recon_loss, discriminator_loss = aae_model.vae.evaluate(x = [pred_p_z, pred_q_z, batch_x], y = batch_x, batch_size = config.batch_size, verbose = 0)
#      print("cost:%f, recon:%f"%(cost, recon_loss))
      avg_cost += 1.*cost / number_of_batches
      avg_recon_cost += 1.*recon_loss / number_of_batches
      avg_discriminator_loss += 1.*  discriminator_loss/ number_of_batches

    # Display logs per epoch step
    if epoch % 1 == 0:
      print("Epoch:%04d, cost=%f, recon_cost:%f, discri_cost:%f"%(epoch + 1, avg_cost, avg_recon_cost, avg_discriminator_loss))  

    #画图
    def plot_latent_variable(epoch):
      output = aae_model.encoder.predict(x_test)
      plt.figure(figsize=(8,8))
      color=plt.cm.rainbow(np.linspace(0,1,10))
      for l,c in zip(range(10),color):
          ix = np.where(dataset[1][1] == l)[0]
          plt.scatter(output[ix,0],output[ix,1],c=c,label=l,s=8,linewidth=0)
      plt.xlim([-5.0,5.0])
      plt.ylim([-5.0,5.0])
      plt.legend(fontsize=15)
      plt.savefig('z_epoch' + str(epoch) + '.pdf')

    if config.latent_dim == 2:
      plot_latent_variable(epoch)


# == CONFIGURATION ==
config = Config()

experiment_repetitions = [1]

# == DATASET ==
#x_train, x_val, x_test, y_train, y_val, y_test = load_dataset( config.dataset_name )

dataset = load_data.load_mnist_full()
x_train,_ = dataset[0]
x_test,_ = dataset[1]

aae_model = AdversarialAutoencoder(config)
# == TRAINING ==
if config.dataset_name == 'mnistDynamic':
  print('***training with dynamic binarization***')
  #x_train = np.random.binomial(1, x_train)
  train()


