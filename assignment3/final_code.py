
'''
name: Sawal Acharya, uni: sa3339
This program implements variational inference method
'''
import numpy
import csv
import numpy.linalg as linalg
import scipy.special as sp
import matplotlib.pyplot as plt


def get_data(loc):
	'''reads the csv_file and returns its contents in the form of an array'''

	all_contents = []
	with open(loc) as csv_file:
		csv_reader = csv.reader(csv_file)
		for row in csv_reader:
			contents = numpy.asarray(row)
			all_contents.append(contents.astype(numpy.float))

	return numpy.asarray(all_contents)

def update_alpha(a, b, u, sigma):
	'''updates the probability distribution on alpha_k'''

	a1 = a + 0.5
	diag = sigma.diagonal()
	diag.shape = u.shape
	b1 = b + 0.5 * (u ** 2 + diag)

	return a1, b1

def update_lambda(e,f, data_mat, y_vect, u, sigma, N, d):
	'''updates the proability distribution on lambda'''

	f1 = 0
	for i in range(N):
		x = data_mat[i,:]
		x.shape = (d,1)
		x_t = numpy.transpose(x)
		t1 = (y_vect[i] - numpy.dot(x_t, u)) ** 2
		t2 = numpy.dot(x_t, numpy.dot(sigma, x))
		f1 = f1 + t1 + t2

	e1 = e + N / 2
	f1 = f + f1 / 2

	return e1, float(f1[0][0])

def update_w(a, b, e, f, data_mat, y):
	'''updates the probability distribution on w'''

	d = data_mat.shape[1]
	n = data_mat.shape[0]
	t1 = 0
	xy = 0
	for i in range(n):
		x = data_mat[i]
		x.shape = (d, 1)
		x_t = numpy.transpose(x)
		t1 += numpy.matmul(x, x_t)
		xy += y[i] * x

	t1 = e / f * t1
	diag_ma = numpy.identity(d) * (a / b)
	sigma1 = linalg.inv(t1 + diag_ma)
	xy = e / f * xy
	u1 = numpy.dot(sigma1, xy)

	return u1, sigma1

def compute_term2(a0, b0, a1, b1, d):
	'''computes the second term of the variation objective function'''

	t1 = 0
	t2 = 0
	for k in range(d):
		t1 = t1 + (sp.digamma(a1[k]) - numpy.log(b1[k]))
		t2 = t2 + a1[k] / b1[k] * (-b0)
	constant = d * (a0 * numpy.log(b0) - sp.gammaln(a0))
	t = constant + (a0 - 1) * t1 + t2
	return t[0]

def compute_term3(e1, f1, X, y, u1, sigma1, n):
	'''computes the third term of the variational objective function'''

	d = X.shape[1]
	constant = - n / 2 * numpy.log(2 * numpy.pi)
	t1 = n / 2 * (sp.digamma(e1) - numpy.log(f1))
	t2 = 0
	for i in range(n):
		x = X[i]
		x.shape = (d, 1)
		x_t = numpy.transpose(x)
		y_i = y[i][0]
		dot1 = (y_i -numpy.dot(x_t, u1)[0][0]) ** 2
		dot2 = numpy.dot(sigma1, x)
		dot3 = numpy.dot(x_t, dot2)[0][0]
		t2 += dot1 + dot3
		#print(dot1, dot3, t1,t2)
	t2 = -e1 / (2 * f1) * (t2)
	#print(t2)
	return t2 + t1 + constant

def compute_term4(a1, b1, u1, sigma, d):
	'''computes the third term of the variational objective function'''

	t1 = 0 
	t2 = 0
	for k in range(d):
		t1 += sp.digamma(a1[k]) - numpy.log(b1[k])
		t2 += a1[k] / b1[k] * (u1[k] ** 2 + sigma[k][k])
	constant = - d / 2 * numpy.log(2 * numpy.pi)

	t = constant + 0.5 * t1 - 0.5 * t2
	return t[0]

def compute_term6(sigma, d):
	'''computes the sixth of variational objective function'''

	#used lodget in order to avoid cases where determinant is 0 or negative
	sign, logdet = numpy.linalg.slogdet(sigma)

	return 0.5 * sign * logdet + d / 2 * numpy.log(2 * numpy.pi) + d / 2

def compute_term7(a1, b1, d):
	'''computes the seventh term of the variational objective function'''

	t1 = 0
	for k in range(d):

		t1 += a1[k] + sp.gammaln(a1[k]) - numpy.log(b1[k]) + (1 - a1[k]) * sp.digamma(a1[k])

	return t1[0]

def compute_variational_objective(a0, b0, e0, f0, a1, b1, e1, f1, u1, sigma1, X, y):
	'''computes the variational objecive function'''

	d = X.shape[1]
	n = X.shape[0]
	term1 = e0 * numpy.log(f0) - sp.gammaln(e0) + (e0 -1) * (sp.digamma(e1) - numpy.log(f1)) - f0 * e1 / f1
	print("term1:",term1)
	term2 = compute_term2(a0, b0, a1, b1, d)
	print("term2:",term2)
	term3 = compute_term3(e1, f1, X, y, u1, sigma1, n)
	print("term3:",term3)
	term4 = compute_term4(a1, b1, u1, sigma1, d)
	print("term4:", term4)
	term5 = sp.digamma(e1) * (1 - e1) + sp.gammaln(e1) + e1 - numpy.log(f1)
	print("term5:",term5)
	term6 = compute_term6(sigma1, d)
	print("term6:",term6)
	term7 = compute_term7(a1, b1, d)
	print("term7:",term7)

	return term1 + term2 + term3 + term4 + term5 + term6 + term7

def run_variational_inference(a0, b0, e0, f0, X, y, z, T):
	'''runs variational inference method'''

	d = X.shape[1] #dimension of the data
	n = X.shape[0] # number of data points
	# remains constant throughout the course of the program
	a_cons = numpy.transpose(numpy.asarray([[a0] * d]))
	b_cons = numpy.transpose(numpy.asarray([[b0] * d]))
	#initialize the variables
	a_ini = numpy.transpose(numpy.asarray([[a0] * d]))
	b_ini = numpy.transpose(numpy.asarray([[b0] * d]))
	e_ini = e0
	f_ini = f0
	u_ini = numpy.zeros((d,1))
	sigma_ini = numpy.identity(d, dtype = 'float64')
	L_all = []
	for t in range(T):
		a_up, b_up = update_alpha(a_cons, b_cons, u_ini, sigma_ini)
		#print("up:",b_up.shape)
		#print(b_up)
		e_up, f_up = update_lambda(e0, f0, X, y, u_ini, sigma_ini, n, d)
		u_up, sigma_up = update_w(a_up, b_up, e_up, f_up, X, y)
		#print(sigma_up)
		a_ini = a_up
		b_ini = b_up
		e_ini = e_up
		f_ini = f_up
		u_ini = u_up
		sigma_ini = sigma_up
		print("iteration_number:",t)
		L = compute_variational_objective(a0, b0, e0, f0, a_up, b_up, e_up,
			f_up, u_up, sigma_up, X, y)

		print("L:",L)
		print()
		L_all.append(L)

	return numpy.asarray(L_all), a_up, b_up, e_up, f_up, u_up

def plot_data(t, L, dataset_name):
	'''plots data of iteration number vs variational objective function'''

	fig1 = plt.figure()
	axes1 = fig1.add_subplot(1,1,1)
	axes1.set_xlabel("t (iteration number)")
	axes1.set_ylabel("Variational objective function")
	axes1.set_title("Variational Objective function for " + dataset_name)
	axes1.plot(t, L)
	fig1.savefig(dataset_name + "_a")

	plt.show()

def plot_expectation_alpha(x_var, y_var, dataset_name ):
	'''plot the inverse of expectation of alpha'''

	fig2 = plt.figure()
	axes2 = fig2.add_subplot(1,1,1)
	axes2.set_xlabel("k")
	axes2.set_ylabel("$E_q[\\alpha_k]$")
	axes2.set_title("plot of " +"$1 / E_q[\\alpha_k]$ " + "for " + dataset_name)
	axes2.plot(x_var, y_var)
	fig2.savefig(dataset_name + "_b")

	plt.show()

def plot_y_vs_z(X, y, z, w, dataset_name):
	'''plots x.w for each data point'''

	y_hat_all = []
	for x in X:
		y_hat = numpy.dot(x, w)
		y_hat_all.append(y_hat)

	fig3 = plt.figure()
	axes3 = fig3.add_subplot(1,1,1)
	axes3.set_xlabel("z")
	axes3.set_ylabel("$\hat{y_i}$")
	axes3.set_title("Plot of $\hat{y_i}$, y, sinc(z) * 10 vs z for " + dataset_name)
	axes3.plot(z, y_hat_all, color = 'blue', label = "$\hat{y_i}$ vs $z_i$")
	axes3.scatter(z, y, color = "green", label = "$y_i$ vs $z_1$")
	axes3.plot(z, numpy.sinc(z) * 10, color = "red", label = "ground truth")
	axes3.legend()
	fig3.savefig(dataset_name + "_d")
	
	plt.show()



location = "/Users/sawal386/Documents/Classes/EECS_6720/"+\
"hw3/data_csv/"

X1 = get_data(location + "X_set1.csv")
X2 = get_data(location + "X_set2.csv")
X3 = get_data(location + "X_set3.csv")
y1 = get_data(location + "y_set1.csv")
y2 = get_data(location + "y_set2.csv")
y3 = get_data(location + "y_set3.csv")
z1 = get_data(location + "z_set1.csv")
z2 = get_data(location + "z_set2.csv")
z3 = get_data(location + "z_set3.csv")
d = X1.shape[1]
n = X1.shape[0]

num_it = 500
x = numpy.linspace(1, num_it, num_it)
result = run_variational_inference(1e-16, 1e-16, 1, 1, X3, y3, z3, num_it)
k = numpy.linspace(1, len(result[1]), len(result[1]))

#part a
plot_data(x, result[0], "Dataset 3")

#part b
plot_expectation_alpha(k, result[2]/result[1], "Dataset 3")

#part c
print("1 \ expectation(lambda):", result[4] / result[3])

#part d
plot_y_vs_z(X3, y3, z3, result[5], "Dataset 3")









