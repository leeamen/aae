#coding:utf8

class Config():
    # == CONFIGURATION ==
    batch_size = 100
    original_dim = 784

    latent_dim = 2

    #先验分布
    z_prior = "uniform"#"gaussian" #"uniform"
    #lamb
    lamb = 10.

    number_epochs = 20
    epsilon_std = 1.0
    learning_rate = 0.0002

    regularization = 'none'

    if regularization == 'none':
        regularization_param = 0
    elif regularization == 'warmup':
        regularization_param = 200
    else:
        raise Exception('Wrong name of regularizer!')

    dataset_name = 'mnistDynamic' #'histopathology'

    data_type = 'binary' #'gray'

    model = 'VAE'

    number_of_flows = 1
