import numpy as np

"""
Using simplex algorithm to solve LP
LP should be in standard form:
max c^T * x
s.t. Ax <= b, x >= 0
-----------------------------------
# add slack variables
# Ax + xs = b
# if b < 0, phase 1 (initialization)
# if b >= 0, phase 2 (simplex)

Here is how a dictionary looks like:
# xs | b    - Ax 
# ---------------------
# y  | c[0] c[1:]^T * x
"""
class SimplexAlgorithm:
	"""
	Main routine of the simplex algorithm. Update the dictionary until it is final.
	"""
	@classmethod
	def simplex(cls, basis, non_basis, b, c, A):
		m, n = len(basis), len(non_basis)
		while True:
			# pick one with maximum coefficient as pivot
			# There are many other rules for selecting the pivot, for more see
			# http://cs.nyu.edu/courses/fall12/CSCI-GA.2945-002/pastproject2.pdf
			pivot = np.argmax(c[1:])
			if c[pivot + 1] <= 0:
				# final dictionary
				return (True, basis, non_basis, b, c, A)
			# find the leaving variable
			max_increment = np.inf
			for i in range(m):
				if b[i] == 0 and A[i, pivot] < 0:
					leave = i
					break
				if A[i, pivot] == 0:
					continue
				increment = b[i] / A[i, pivot] 
				# we choose the variable with lowest possible increment as the leaving variable
				# prioritize x0 in the initialization phase
				if increment > 0 and (increment < max_increment or 
					(increment <= max_increment and basis[i] == 0)):
					max_increment = increment
					leave = i
			if max_increment is np.inf:
				# unbounded problem
				return (False, basis, non_basis, b, c, A)
			# update dictionary
			basis, non_basis, b, c, A = cls.update(basis, non_basis, b, c, A, pivot, leave)


	"""
	Update the dictionary given entering and leaving index.
	"""
	@classmethod
	def update(cls, basis, non_basis, b, c, A, pivot, leave):
		# update leave-th row of matrix A
		coeff = A[leave, pivot]
		A[leave, :] /= coeff
		A[leave, pivot] = 1.0 / coeff
		# update leave-th row of vector b
		b[leave] /= coeff
		# update other rows of matrix A and vector b
		for i in range(len(basis)):
			if i == leave:
				continue
			b[i] += - A[i, pivot] * b[leave]
			temp = - A[i, pivot] * A[leave, pivot] 
			A[i, :] += - A[i, pivot] * A[leave, :]
			A[i, pivot] = temp
		# update vector c
		c[0] += c[pivot + 1] * b[leave]
		temp = - A[leave, pivot] * c[pivot + 1]
		c[1:] += - A[leave, :] * c[pivot + 1]
		c[pivot + 1] = temp
		# update basis and non_basis variables
		basis[leave], non_basis[pivot] = non_basis[pivot], basis[leave]
		return (basis, non_basis, b, c, A)

	"""
	Recover the optimal solution and optimal value from a simplex algorithm.
	"""
	@classmethod
	def solution(cls, basis, non_basis, b, c):
		m, n = len(basis), len(non_basis)
		y = c[0]
		x = [0.0] * n
		for i in range(m):
			if basis[i] <= n:
				x[basis[i] - 1] = b[i]
		return x, y

	"""
	Initialization phase of simplex algorithm.
	"""
	@classmethod
	def initialization(cls, basis, non_basis, b, c, A):
		m, n = len(basis), len(non_basis)
		if np.any(b < 0):
			# save original objective function
			objective = zip(non_basis, c[1:])
			# build the auxiliary problem
			non_basis.append(0)
			A = np.append(A, [[-1]] * m, axis = 1)
			c = [0] * (n + 1) + [-1]
			# choose x0 as the enter variable
			pivot = n
			# choose the variable with minimum b as leaving variable
			leave = np.argmin(b)
			# update dictionary
			basis, non_basis, b, c, A = cls.update(basis, non_basis, b, c, A, pivot, leave)
			# solve the auxiliary problem
			flag, basis, non_basis, b, c, A = cls.simplex(basis, non_basis, b, c, A)
			# the auxiliary problem must be feasible and bounded, therefore flag = True
			assert flag
			if c[0] != 0:
				# optimal value of auxiliary problem is not 0, therefore original problem is infeasible
				return (False, basis, non_basis, b, c, A)
			else:
				assert np.all(b >= 0)
				# build starting dictionary for the original problem
				index_x0 = non_basis.find(0)
				A = np.delete(A, index_x0, axis = 1)
				c = np.array([0] * n)
				for var, coeff in objective:
					if var in basis:
						i = basis.index(var)
						c += - coeff * A[i, :]
					else:
						i = non_basis.index(var)
						c[i] += coeff
				np.insert(c, 0, 0)
				return (True, basis, non_basis, b, c, A)
		else:
			# original problem is feasible, so nothing to do at initialization phase
			return (True, basis, non_basis, b, c, A)

	"""
	Solve the LP.
	
	b - numpy array of size (m,)
	c - numpy array of size (n,)
	A - numpy array of size (m, n)
	"""
	@classmethod
	def solve(cls, A, b, c):
		c = np.insert(c, 0, 0) # internally we use c[0] to store the value of objective function
		m, n = len(b), len(c) - 1
		# x1, x2, x3, ..., xn   (n variables)
		# x_{n+1}, x_{n+2}, ..., x_{m+n} (m slack variables)
		non_basis = range(1, n + 1)
		basis = range(n + 1, m + n + 1)

		flag, basis, non_basis, b, c, A = cls.initialization(basis, non_basis, b, c, A)
		if not flag:
			print "LP infeasible!"
			return None, None
		flag, basis, non_basis, b, c, A = cls.simplex(basis, non_basis, b, c, A)
		if not flag:
			print "LP unbounded!"
			return None, None
		return cls.solution(basis, non_basis, b, c)

# Examples
m = 20
n = 100
c = np.random.rand(n)
b = np.random.rand(m)
A = np.random.rand(m, n)
print SimplexAlgorithm.solve(A, b, c)


