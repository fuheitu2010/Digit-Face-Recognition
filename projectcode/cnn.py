
# CNN implementation
import util
import numpy as np
import cPickle as pickle
import theano
from lasagne import layers
from nolearn.lasagne import NeuralNet

class AdjustVariable(object):
    def __init__(self, name, start=0.03, stop=0.001):
        self.name = name
        self.start, self.stop = start, stop
        self.ls = None

    def __call__(self, nn, train_history):
        if self.ls is None:
            self.ls = np.linspace(self.start, self.stop, nn.max_epochs)

        epoch = train_history[-1]['epoch']
        new_value = np.cast['float32'](self.ls[epoch - 1])
        getattr(nn, self.name).set_value(new_value)

class CnnClassifier:
  """
  CNN classifier.
  
  """

  def __init__( self, legalLabels, max_iterations, use_training):

    self.legalLabels = legalLabels
    self.type = "cnn"
    self.num_epoches = max_iterations
    self.learning_rate = 0.01
    self.momentum = 0.9
    self.batch_size = 600
    self.use_training = use_training

  def pickle_load(self,f, encoding):
    return pickle.load(f)

  def initializeWeightsToZero(self):
    "Resets the weights of each label to zero vectors"
    self.weights = {}
    for label in self.legalLabels:
      self.weights[label] = util.Counter() # this is the data-structure you should use

  def reshape(self, Y, dim):
    "from 1 column to dim columns"

    Y_new = np.zeros((Y.shape[0], dim))
    for i in range(Y.shape[0]):
      Y_new[i][Y[i]] = 1
    return Y_new

  def reshape_inv(self, Y, dim):
    "from dim columns to 1 column"
    Y_new = np.zeros(Y.shape[0])
    for i in range(Y.shape[0]):
      Y_new[i] = np.argmax(Y[i])
    return Y_new

  def load(self, trainingData, trainingLabels, validationData, validationLabels):

      dim = len(set(self.legalLabels))
      X_train = np.array(trainingData)
      X_train = X_train.astype(np.float32)
      y_train = np.array(trainingLabels)
      y_train_reshape = self.reshape(y_train, dim)
      y_train_reshape = y_train_reshape.astype(np.float32)
      X_valid = np.array(validationData)
      X_valid = X_valid.astype(np.float32)
      y_valid = np.array(validationLabels)
      y_valid_reshape = self.reshape(y_valid, dim)
      y_valid_reshape = y_valid_reshape.astype(np.float32)

      return dict(
        X_train = X_train,
        y_train = y_train_reshape,
        X_valid = X_valid,
        y_valid = y_valid_reshape,
        num_examples_train=X_train.shape[0],
        num_examples_valid=X_valid.shape[0],
        input_height = X_train.shape[2],
        input_width = X_train.shape[3],
        output_dim = dim,
        )

  def build_model(self, input_width, input_height, output_dim):
      self.net = NeuralNet(
      layers=[
        ('input', layers.InputLayer),
        ('conv1', layers.Conv2DLayer),
        ('pool1', layers.MaxPool2DLayer),
        ('conv2', layers.Conv2DLayer),
        ('pool2', layers.MaxPool2DLayer),
        ('hidden1', layers.DenseLayer),
        ('dropout2', layers.DropoutLayer),
        ('output', layers.DenseLayer),
        ],
      input_shape=(None, 1, input_height, input_width),
      conv1_num_filters=32, conv1_filter_size=(5, 5), pool1_pool_size=(2, 2),
      conv2_num_filters=32, conv2_filter_size=(5, 5), pool2_pool_size=(2, 2),
      hidden1_num_units=256,
      dropout2_p = 0.5,
      output_num_units=output_dim,
      output_nonlinearity=None,
      update_learning_rate=theano.shared(np.cast['float32'](0.01)),
      update_momentum=theano.shared(np.cast['float32'](0.9)),

      regression=True,
      on_epoch_finished=[
        AdjustVariable('update_learning_rate', start=np.cast['float32'](0.03), stop=0.0001),
        AdjustVariable('update_momentum', start=np.cast['float32'](0.9), stop=0.999),
        ],
      max_epochs=self.num_epoches,
      verbose=1,
      )

  def train(self, trainingData, trainingLabels, validationData, validationLabels):
    "Outside shell to call your method. Do not modify this method."

    if not self.use_training:
      return

    dataset = self.load(trainingData, trainingLabels, validationData, validationLabels)
    self.build_model(input_height=dataset['input_height'],input_width=dataset['input_width'],output_dim=dataset['output_dim'])
    self.net.fit(dataset['X_train'],dataset['y_train'])

    with open('net.pickle', 'wb') as f:
        pickle.dump(self.net, f, -1)

  def classify(self, data):
    """
    Classifies each datum as the label that most closely matches the prototype vector
    for that label.  See the project description for details.
    """
    dim = len(set(self.legalLabels))
    if self.use_training:
        net = self.net
    else:
      if dim == 10:
        with open('net_digits.pickle','rb') as f:
          net = pickle.load(f)
      else:
        with open('net_faces.pickle','rb') as f:
          net = pickle.load(f)

    X_test = np.array(data)
    X_test = X_test.astype(np.float32)
    guesses = net.predict(X_test)
    guesses_reshape = self.reshape_inv(guesses, dim)
    guesses_reshape = guesses_reshape.astype(np.int64)
    guess_list = guesses_reshape.tolist()

    return guess_list
