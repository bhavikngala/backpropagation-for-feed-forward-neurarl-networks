from helpers import helperFunctions as misc
from backprop import BackPropNN
from tensorflow.examples.tutorials.mnist import input_data

def main():
	backpropnetwork = BackPropNN([784, 100, 10])
	backpropnetwork.describeNetwork()

	mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

	print('\n\n~~~~~~~Starting training ~~~~~~~')
	backpropnetwork.train(mnist.train.images, mnist.train.labels, 0.5, 1000, 100, mnist.validation.images, mnist.validation.labels)
	print('\n~~~~~~~Training completed~~~~~~~\n\n')

	print('classification error on test data:', backpropnetwork.evaluateNetwork(mnist.test.images, mnist.test.labels))

	# normalize the input
	# misc.normalize(x)

if __name__ == '__main__':
	main()