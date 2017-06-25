

from theano.tensor.shared_randomstreams import RandomStreams
import numpy as np,theano
import theano.tensor as T

rng = RandomStreams(seed=np.random.randint(1234))
def get_normalized_vector(v):
    v = v / (1e-20 + T.max(T.abs_(v), axis=1, keepdims=True))
    v_2 = T.sum(v**2,axis=1,keepdims=True)
    return v / T.sqrt(1e-6+v_2)

def sample_from_prior(z, z_prior):

    ###### gausssian #######
    if(z_prior is 'gaussian'):
        return (1.0*rng.normal(size=z.shape,dtype=theano.config.floatX)).eval()

    ###### uniform ########
    elif(z_prior is 'uniform'):
        v = get_normalized_vector(rng.normal(size=z.shape,dtype=theano.config.floatX))
        #print(v.eval())
        r = T.power(rng.uniform(size=z.sum(axis=1,keepdims=True).shape,low=0,high=1.0,dtype=theano.config.floatX),1./z.shape[1])
        #print(r.eval())
        r = T.patternbroadcast(r,[False,True])
        #print(r.eval())
        return (2.0*r*v).eval()

    else:
        raise NotImplementedError()

if __name__ == "__main__":
  
  a = np.array([[1.2, 3, 4.5],[2,3,5],[7,6,8]])
  z_prior = "uniform"
  p_z = sample_from_prior(a, z_prior)
  
  print(p_z)
