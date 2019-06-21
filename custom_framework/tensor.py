import numpy as np

class Tensor(object):

	def __init__(self, data, autograd=False, creators=None, creation_op=None, id=None):
		self.data = np.array(data)

		self.autograd = autograd
		self.children = {}

		# A list of any tensors used in the creation of this tensor
		self.creators = creators

		# The type of operation used to create this operator
		self.creation_op = creation_op

		# The gradient of the output w.r.t. this tensor
		self.grad = None

		# Generate a unique ID 
		if id is None:
			id = np.random.randint(0, 100000)
		self.id = id

		# Keep track of how many children this tensor has
		if creators is not None:
			for c in creators:
				if self.id not in c.children:
					c.children[self.id] = 1
				else:
					c.children[self.id] += 1

	def __add__(self, other):
		return Tensor(self.data + other.data, creators=[self, other], creation_op='add')

	def __repr__(self):
		return str(self.data.__repr__())

	def __str__(self):
		return str(self.data.__str__())

	def backward(self, grad):
		self.grad = grad

		if self.creation_op == 'add':
			self.creators[0].backward(grad)
			self.creators[1].backward(grad)

x = Tensor([1, 2, 3, 4, 5])
y = Tensor([2, 2, 2, 2, 2])
z = x + y
z.backward(Tensor(np.array([1, 1, 1, 1, 1])))

print(z)
print(x.grad)
print(y.grad)
print(z.creators)
print(z.creation_op)