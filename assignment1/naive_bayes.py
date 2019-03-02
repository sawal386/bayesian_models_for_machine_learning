#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 22 15:17:46 2018

@author: sawal386

This program uses Naive-Bayes method to classify emails as spam/non-spam
"""

import csv
import numpy as np
import math
import scipy.misc as misc
import scipy.stats as stat
import pandas
import matplotlib.pyplot as plt

def get_data(filename):
    '''reads the csv file(filename) and returns its contents in the form 
       of arrays'''
       
    all_data = []
    with open(filename) as source:
        reader = csv.reader(source)
        for row in reader:
            all_data.append(row)
    
    all_data = np.asarray(all_data)
    #convert the numbers from string to integers
    array_int = all_data.astype(np.int)
    
    return array_int

def get_data_specified_label(X, y, tag):
    '''returns an array whose elements have the same label as tag
       (variable name)'''
    
    data_label_list = []
    N = X.shape[0]
    count = 0
    for i in range(N):
        if y[i] == tag:
            data_label_list.append(X[i])
            count +=1 
    data_label_arr = np.asarray(data_label_list)
    #print(data_label_arr.shape)
    
    return data_label_arr

def compute_predictive_value(X, y, label, alpha, beta, x_star):
    '''returns the logarithm of P(y_star = y|x*, X, y) i.e conditional 
       probability of the label of new data being equal to a given value'''

    num_features = train_data.shape[1]
    data = get_data_specified_label(X, y, label)
    #print(data.shape)
    n = data.shape[0]
    data_sum = np.sum(data, axis = 0)
    overall_predicted_value = 0
    for d in range(num_features):
        p = 1 /(n + beta + 1)
        k = x_star[d]
        r = alpha + data_sum[d]
        q = 1 - p
        #taking the logarithm of the probability
        predicted_value_d = np.log(misc.comb(r + k - 1, k)) + k * np.log(p) + r * np.log(q)
        overall_predicted_value = overall_predicted_value + predicted_value_d 

    return overall_predicted_value

def test_results(obtained_values, true_values):
    '''computes the testing error and accuracy, and present the results 
       in a confusion matrix'''
    
    n_spam_correct = 0
    n_spam_incorrect = 0
    n_nspam_correct = 0
    n_nspam_incorrect = 0
    n_mistake = 0
    n = len(obtained_values)
    misclassified_indices = []
    for i in range(n):
        #print('true:',true_values[i], "obtained:", obtained_values[i])
        if true_values[i] == 0:
            if obtained_values[i] == 0:
                n_nspam_correct += 1
            else:
                n_nspam_incorrect += 1
                misclassified_indices.append(i)
        else:
            if obtained_values[i] == 1:
                n_spam_correct += 1
            else:
                n_spam_incorrect += 1
                misclassified_indices.append(i)
    
    heading2 = ["spam", "not spam"]
    heading1 = ["classified as spam", "classified as non-spam"]
    frequency = np.asanyarray([[n_spam_correct, n_spam_incorrect], 
                               [n_nspam_incorrect, n_nspam_correct]])
    table = pandas.DataFrame(frequency,heading2, heading1)
    print()
    print("confusion matrix")
    print(table)
    
    return misclassified_indices

  
def produce_plot(array, X_train, X_test, y_train, y_test, alpha, beta, feat,
                    name, question):
    '''produce plots to graphically illustrate the mean value of the feature'''
    
    spam_data = get_data_specified_label(X_train, y_train, 1)
    nspam_data = get_data_specified_label(X_train, y_train, 0)  
    sum_spam_data = np.sum(spam_data, axis = 0)
    sum_nspam_data = np.sum(nspam_data, axis = 0) 
    n_spam = spam_data.shape[0]
    n_nspam = nspam_data.shape[0]
    expectation_1 = (sum_spam_data + alpha) / (n_spam + b)
    expectation_0 = (sum_nspam_data + alpha) / (n_nspam + b)
    for i in range(3):
        fig1 = plt.figure()
        axes1 = fig1.add_subplot(1,1,1)
        axes1.set_xlabel("features")
        axes1.set_ylabel("$E[\lambda_i]$")
        axes1.set_title(question + " Plot for "+ name)
        index = array[i]
        y = X_test[index]
        x = np.linspace(1,54,54)

        axes1.scatter(x, expectation_1, label = "$E[\lambda_1]$ ")
        axes1.scatter(x, expectation_0, label = "$E[\lambda_0]$ ")
        axes1.scatter(x, y, label = "data points for testing data #"+str(index))
        axes1.set_xticks(x)
        axes1.set_xticklabels(feat, rotation = "vertical")
        axes1.legend()
    
    plt.show()

location = "/Users/sawal386/Documents/Classes/EECS_6720/hw1_data/"
train_label_file = location + 'label_train.csv'
train_data_file = location + "X_train.csv"
test_label_file = location + "label_test.csv"
test_data_file = location + "X_test.csv"

train_data = get_data(train_data_file)
test_data = get_data(test_data_file)
train_label = get_data(train_label_file)
test_label = get_data(test_label_file)

#constants
a = 1
b = 1
e = 1
f = 1
N = train_data.shape[0]
num_spam_train = np.sum(train_label)
num_nspam_train =  N - num_spam_train

#P(y* = 1|y)
p_y1 = (e + num_spam_train) / (N + e + f)
#P(y* = 0|y)
p_y0 = (f + num_nspam_train)/ (N + e + f)


count = 0
classifier_results_li = []
ambiguous_index = []
calculated_probabilities_li = []

#compute the predictive probabilities
for points in test_data:
    xstar = points
    p_xstar_1 = compute_predictive_value(train_data,train_label, 1, a, b, 
                                     xstar) + np.log(p_y1)
    p_xstar_0 = compute_predictive_value(train_data, train_label, 0, a, b, 
                                     xstar) + np.log(p_y0)
    p1 = np.exp(p_xstar_1)/(np.exp(p_xstar_1) + np.exp(p_xstar_0))
    p0 = np.exp(p_xstar_0)/(np.exp(p_xstar_1) + np.exp(p_xstar_0))
    prob = [p0,p1]
    calculated_probabilities_li.append(prob)
    #print(p1, p0)
    if abs(0.5 - p1) <= 0.05:
        ambiguous_index.append(count)
    if p_xstar_1 > p_xstar_0:
        classifier_results_li.append(1)
        #print("1")
    else:
        #print("0")
        classifier_results_li.append(0)
    count += 1
    
result = np.asarray(classifier_results_li)
mistakes_index = test_results(result, test_label)
calculated_probabilities = np.asarray(calculated_probabilities_li)

features = ['make', 'address', 'all', '3d', 'our', 'over', 'remove', 
            'internet', 'order', 'mail', 'receive', 'will', 'people', 'report',
            'addresses', 'free', 'business', 'email', 'you', 'credit', 'your', 
            'font', '000', 'money', 'hp', 'hpl', 'george', '650', 'lab', 
            'labs', 'telnet', '857', 'data', '415', '85', 'technology', '1999',
            'parts', 'pm', 'direct', 'cs', 'meeting', 'original', 'project', 
            're', 'edu', 'table', 'conference', ';', '(', '[', '!', '$', '#']


produce_plot(mistakes_index, train_data, test_data, train_label, test_label, 
             a, b, features, "mistaken results","Problem 4 part c")
produce_plot(ambiguous_index, train_data, test_data, train_label, test_label, 
             a, b, features, "ambiguous results","Problem 4 part d")


    
    
    
    
    


                                 
    








