import csv
import numpy
from scipy.stats import norm
import numpy.linalg as linalg
import matplotlib.pyplot as plt
import pandas

def get_data(loc):
	'''reads the csv_file and returns its contents in the form of an array'''

	all_contents = []
	with open(loc) as csv_file:
		csv_reader = csv.reader(csv_file)
		for row in csv_reader:
			contents = numpy.asarray(row)
			all_contents.append(contents.astype(numpy.int))

	return numpy.asarray(all_contents)

def get_data_matrix(shape, raw_data):
	'''returns a matrix of given shape. The rows and columns correspond
	to user_id and movies_id respectively'''

	matrix = numpy.zeros(shape)
	for row in raw_data:
		user_id = row[0] - 1
		movies_id = row[1] - 1
		rating = row[2]
		matrix[user_id][movies_id] = rating 

	return matrix

def get_expectation_matrix(U, V, r_matrix, s):
	'''returns the expectation matrix i.e. the expected value of the
	probability distribution on the latent variable'''
	
	E = []
	dot_UV = numpy.dot(U,numpy.transpose(V))
	pdf = norm.pdf((1) * dot_UV / s)
	cdf = norm.cdf((-1) * dot_UV / s)
	for i in range(r_matrix.shape[0]):
		e_i = []
		for j in range(r_matrix.shape[1]):
			r_ij = r_matrix[i][j]
			e = 0
			if r_ij == 1:
				e = dot_UV[i][j] + s * pdf[i][j] / (1 - cdf[i][j])
			elif r_ij == -1:
				e = dot_UV[i][j] - s * pdf[i][j] / cdf[i][j]
			e_i.append(e)
		E.append(e_i)

	return numpy.asarray(E)

def update_U(c, varia, d, size, V, E):
	'''updates U vector. part of the M step of EM Algorithm'''

	U_update = []
	V_trans = numpy.transpose(V)
	# sum of v.v_tranpose for all values of v(from 1 to M)
	sum_V_dot_all = numpy.dot(V_trans,V)
	a = 1 / c * numpy.identity(d) + 1 / varia * sum_V_dot_all
	for i in range(size):
		#ith row of the expectation matrix
		e = E[i]
		e.shape = (len(V), 1)
		b = 1 / varia *  numpy.dot(V_trans, e)
		u_i = numpy.dot(linalg.inv(a), b)
		u_i.shape = (d,)
		U_update.append(u_i)

	return numpy.asarray(U_update)

def update_V(c, varia, d, size, U, E):
	'''updates V. part  the M step of EM algorithm '''
	
	V_update = []
	U_trans = numpy.transpose(U)
	# sum of u.u_tranpose for all values of u(from 1 to n)	
	sum_U_dot_all = numpy.dot(U_trans, U)
	sum_u = 0
	for u in U:
		u.shape = (d,1)
		sum_u += numpy.dot(u,numpy.transpose(u))
	a = 1 / c * numpy.identity(d) + 1 / varia * sum_U_dot_all
	for j in range(size):
		e = E[:,j]
		e.shape = (len(U), 1)
		b = 1 / varia * numpy.dot(U_trans, e)
		v_i = numpy.dot(linalg.inv(a), b)
		v_i.shape = (d,)
		V_update.append(v_i)

	return numpy.asarray(V_update)

def compute_sum(X):
	'''computes the sum of dot product of all the row vectors in X''' 
	sum_x = 0
	sum_y = 0
	for x in X:
		sum_x += numpy.dot(numpy.transpose(x), x)

	return sum_x

def compute_l_ruv(data, U, V, d, c, sigma):
	'''computes the joint probability distribution p(R,U,V)'''
	
	sum_cons = -(U.shape[0] + V.shape[0]) * d / 2 * (numpy.log(2 * numpy.pi * c))
	# log(p(u)), multivariate distrbituion
	sum_u = -1 / (2 * c) * compute_sum(U)
	#log(p(v))
	sum_v = -1 / (2 * c) * compute_sum(V)
	dot_UV = numpy.dot(U, numpy.transpose(V))
	cdf_UV = norm.cdf(dot_UV / sigma)
	ln_r_uv = 0
	for i in range(len(U)):
		for j in range(len(V)):
			r_ij = data[i][j]
			if r_ij == 1:
				ln_r_uv += numpy.log(cdf_UV[i][j])
			elif r_ij == -1:
				ln_r_uv += numpy.log(1 - cdf_UV[i][j])
	print("sum_u",sum_u, "sum_v:",sum_v, "sum_cons:",sum_cons, "p(R|UV):",ln_r_uv)
	return sum_cons + sum_u + sum_v + ln_r_uv


def run_em(matrix, T, dim, varia, c):
	'''runs EM algorithm for upto T iterations and returns the result'''

	result = []
	U_size = matrix.shape[0]
	V_size = matrix.shape[1]
	#mean and variance for u_0 and v_0
	mean_ini = numpy.zeros(5)
	variance_ini = 0.1 * numpy.identity(5)
	U_0 = numpy.random.multivariate_normal(mean_ini, variance_ini, U_size)
	V_0 = numpy.random.multivariate_normal(mean_ini, variance_ini, V_size)
	for t in range(100):
		print("t:", t)
		E_q = get_expectation_matrix(U_0, V_0, matrix, varia ** 0.5)
		U_1 = update_U(c, varia, dim, U_size, V_0, E_q)
		V_1 = update_V(c, varia, dim, V_size, U_1, E_q)
		l_ruv = compute_l_ruv(matrix, U_1, V_1, dim, c, varia ** 0.5)
		U_0 = U_1
		V_0 = V_1
		print("l_ruv:",l_ruv)
		print()
		result.append(l_ruv)
	return (numpy.asarray(result), U_0, V_0)

def run_2a(data, T, d, v, c):
	'''runs part 2a of the assignment'''

	final_result = run_em(data, T, d, v, c)
	ln_p_ruv = final_result[0]
	t = numpy.linspace(2,100,99)
	fig1 = plt.figure()
	axes1 = fig1.add_subplot(1,1,1)
	axes1.set_xlabel("t (iteration number")
	axes1.set_ylabel("ln p(R,U,V)")
	axes1.set_title("plot of ln p(R,U,V) for iterations 2 to 100")
	axes1.plot(t, numpy.asarray(ln_p_ruv)[1:])

	plt.show()

	U = final_result[1]
	V = final_result[2]
	
	return (U, V)

def run_2b(data, T, d, v, c, n):
	'''runs part 2a 5 times, i.e. for five different starting values'''

	all_results = []
	for i in range(5):
		result = run_em(data, T, d, v, c)[0]
		all_results.append(result)


	fig2 = plt.figure()
	axes2 = fig2.add_subplot(1,1,1)
	axes2.set_xlabel("t (iteration number")
	axes2.set_ylabel("ln p(R,U,V)")
	axes2.set_title("plot of ln (R,U,V) for iterations 20 through 100")
	t = numpy.linspace(20,100,81)
	axes2.plot(t,all_results[0][19:], label = "1")
	axes2.plot(t,all_results[1][19:], label = "2")
	axes2.plot(t,all_results[2][19:], label = "3")
	axes2.plot(t,all_results[3][19:], label = "4")
	axes2.plot(t,all_results[4][19:], label = "5")
	axes2.legend()

	plt.show()

def run_2c(U, V, loc):

	'''runs part 2c of the assignment, where we chech the accuracy of 
	   the method'''

	test_data = get_data(loc + "ratings_test.csv")
	true_positive = 0
	false_positive = 0
	true_negative = 0
	false_negative = 0
	sigma = variance ** 0.5
	dot_UV_updated = numpy.dot(U, numpy.transpose(V))
	total_cases = 0

	for row in test_data:
		user_id = row[0] - 1
		movie_id = row[1] - 1
		true_rating = row[2]
		classification = 0
		u = U[user_id]
		v = V[movie_id]
		p_1 = norm.cdf(dot_UV_updated[user_id][movie_id]/ sigma)
		p_0 = 1 - p_1

		if p_1 > p_0:
			classification = 1
		else:
			classification = -1

		if true_rating == 1 and classification == 1:
			true_positive += 1
		elif true_rating == 1 and classification == -1:
			false_negative += 1
		elif true_rating == -1 and classification == 1:
			false_positive += 1
		elif true_rating == -1 and classification == -1:
			true_negative += 1
		total_cases += 1
	accuracy = (true_positive + true_negative) / total_cases
	print(accuracy)
	print("positive correct:", true_positive, "positive incorrect:",false_negative,
		"negative correct:", true_negative, "negative incorrect:", false_positive)

	heading2 = ["postive (1)", "negative spam (-1)"]
	heading1 = ["classified as positive", "classified as negative"]
	frequency = numpy.asanyarray([[true_positive, false_negative], 
                               [false_positive, true_negative]])
	table = pandas.DataFrame(frequency,heading2, heading1)
	print("Confusion matrix")
	print(table)



#location of data_files
location = "/Users/sawal386/Documents/Classes/EECS_6720/"+\
"hw2/movies_csv/"
filename = "ratings.csv"

# constants
d = 5
variance = 1
c = 1
movies_data = get_data(location + filename)
num_users = numpy.amax(movies_data[:,0])
num_movies = numpy.amax(movies_data[:,1])
data_matrix = get_data_matrix((num_users, num_movies), movies_data)
U, V = run_2a(data_matrix, 100, d, variance, c)
run_2b(data_matrix, 100, d, variance, c, 5)
run_2c(U,V, location)









