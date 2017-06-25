#coding:utf8

import matplotlib
#保存图片
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#%matplotlib inline

import os
import keras.backend as K
from aae import AdversarialAutoencoder
from utils.configuration import Config
import numpy as np

from utils.load_data import load_dataset

#os.environ["CUDA_VISIBLE_DEVICES"]="1"

np.random.seed(100)
#tf.set_random_seed(0)


# == CONFIGURATION ==
config = Config()


# == DATASET ==
x_train, x_val, x_test, y_train, y_val, y_test = load_dataset( config.dataset_name )

aae_model = AdversarialAutoencoder(config)
aae_model.vae.load_weights(str(config.latent_dim) + "ave_weights.hdf5")

# == TRAINING ==
if config.dataset_name == 'mnistDynamic':
  print('***training with dynamic binarization***')

  #使用测试集生成一些图片对比一下
  batch_id = 5
  x_sample = x_test[batch_id * config.batch_size : config.batch_size * (batch_id + 1), :]
  p_z_y = np.array([[0.]] * config.batch_size, dtype = np.float32)
  x_reconstruct = aae_model.vae.predict([p_z_y, p_z_y, x_sample], batch_size = config.batch_size)
  print("x_reconstruct", x_reconstruct)

  plt.figure(figsize=(8, 12))
  for i in range(5):
      plt.subplot(5, 2, 2*i + 1)
      plt.imshow(x_sample[i].reshape(28, 28), vmin=0, vmax=1, cmap = "gray")
      plt.title("Test input")
      plt.colorbar()
      plt.subplot(5, 2, 2*i + 2)
      plt.imshow(x_reconstruct[i].reshape(28, 28), vmin=0, vmax=1, cmap="gray")
      plt.title("Reconstruction")
      plt.colorbar()
  #plt.tight_layout()
  plt.savefig('1.jpg') 


  '''
  再次训练, 改变z的节点个数
  画流图 mainfold
  为了显示出上边不同颜色的点代表的聚类结果
  '''
  x_sample = x_test
  y_sample = y_test
  print(y_sample)
  print(y_sample.shape)

  z_mu = aae_model.encoder.predict(x_sample)

  plt.figure(figsize=(8, 6))
  plt.scatter(z_mu[:, 0], z_mu[:, 1], c=y_sample)
  plt.colorbar()
  plt.grid()
  plt.savefig('2.jpg')

  """
  流图
  """
  nx = ny = 20
  x_values = np.linspace(-3, 3, nx)
  y_values = np.linspace(-3, 3, ny)
  
  #像素28, 一个方向放20个图片
  canvas = np.empty((28*ny, 28*nx))
  #enumerate, 前面带个编号: (1, -2.6842105263157894)
  for i, yi in enumerate(x_values):
      for j, xi in enumerate(y_values):
          z_mu = np.array([[xi, yi] * (config.latent_dim/2)])
          x_mean = aae_model.decoder.predict(z_mu)
          #batch_size个图片，只取第一个画出来
          canvas[(nx-i-1)*28:(nx-i)*28, j*28:(j+1)*28] = x_mean[0].reshape(28, 28)
  
  plt.figure(figsize=(8, 10))
  Xi, Yi = np.meshgrid(x_values, y_values)
  plt.imshow(canvas, origin="upper", cmap="gray")
  plt.savefig('3.jpg')
 
