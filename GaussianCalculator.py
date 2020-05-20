from math import pi
from math import e
from math import floor
import numpy
import random

def m(o):
    if isinstance(o[0],list):
        new_m = []
        for i in range(len(o[0])):
            new_m.append(sum([x[i] for x in o])/len(o))
        return new_m
    else:
        return sum(o)/len(o)

def s(m,o):
    if isinstance(m,list):
        new_s = []
        if len(o)!=1:
            for k in range(len(o[0])):
                try:
                    new_s.append(sum([(dev[k] - m[k])**2 for dev in o])/len(o))
                except:
                    raise ValueError('Are observations a list of lists? They should be.')
            return new_s
        else:
            for k in range(len(o[0])):
                new_s.append(0.00001)
            return new_s
    else:
        return sum([(m-dev)**2 for dev in o])/len(o)

def pdf(m,s,o):
    if isinstance(m,list) or type(m) is numpy.ndarray:
        # Compute pdf for a multivariate gaussian with diagonal covariance
        new_m = numpy.array(m)
        new_s = numpy.array(s)
        prob = 1
        for d in range(len(o)):
            prob *= (1/((2*pi*s[d])**(1/2)))*e**(-1/2*((o[d]-m[d])**2/(s[d])))
        return prob
    else:
        return 1/(2*pi*s)**(1/2)*e**-((o-m)**2/(2*s))

def euclidean_distance(o,mu):
    eucl = []
    for index, element in enumerate(o):
        if isinstance(element, list):
            assert len(element)==len(mu[index]), 'Vector {} has different number of dimensions from mean {}'.format(element,mu[index])
            eucl.append(sum([(element[i]-mu[index][i])**2 for i in range(len(element))])**(1/2))
        else:
            return (sum([(o[i]-mu[i])**2 for i in range(len(o))]))**(1/2)
    return (eucl)

def k_mean(o,k):
    dim = len(o[0])
    # print(dim)
    no_stop = True
    # UNIFORM INITIALISATION
    modulo = len(o) % k
    division = floor(len(o)/k)
    grouped_k = [o[i*division:(1+i)*division] for i in range(k)]
    if modulo!=0:
        while modulo>0:
            # print(grouped_k[0])
            try:
                grouped_k[0].append(o[-modulo])
            except AttributeError:
                numpy.append(grouped_k[0], o[-modulo])
            modulo-=1
    k_means = numpy.array([m(grouped_k[i]) for i in range(len(grouped_k))], dtype='float64')
    # BELOW CODE: RANDOM INITIALISATION
    # k_means = numpy.array([])
    # for x in range(k):
    #     if len(k_means) == 0:
    #         k_means = numpy.array([random.randrange(numpy.min(o[:,d]),numpy.max(o[:,d])+1) for d in range(dim)],dtype='float64')
    #     else:
    #         k_means = numpy.vstack([k_means, [random.randrange(numpy.min(o[:,d]),numpy.max(o[:,d])+1) for d in range(dim)]])
    # print(k_means)
    w = 1
    while no_stop:
        subsets = {key: [] for key in range(k)}
        for obs in o:
            best = 0
            for index, mean in enumerate(k_means):
                dist = euclidean_distance(obs,mean)
                if isinstance(best, int):
                    best = [index,dist]
                elif dist < best[1]:
                    best = [index, dist]
            subsets[best[0]].append(obs)
        # print(subsets)
        same_means = 0
        for key,value in subsets.items():
            try:
                new_mean = m(value)
            except IndexError:
                k_means[key] = numpy.array(
                    [random.randrange(numpy.min(o), numpy.max(o) + 1) for d in range(dim)], dtype='float64')
                continue
            if (k_means[key] == new_mean).all():
                same_means += 1
            k_means[key] = new_mean
        if same_means == k:
            no_stop = False
        w+=1
    return k_means, subsets


def GMM(o, m, fixed_iterations=False):
    init = k_mean(o,m)
    gaussian_means = []
    gaussian_variances = []
    for index,mu in enumerate(init[0]):
        gaussian_means.append(mu)
        gaussian_variances.append(s(mu,init[1][index]))
    previous_gaussian_means = gaussian_means.copy()
    component_prob = [len(init[1][i])/len(o) for i in range(m)]
    prova = 0
    counter = 0
    while True:
        component_occupation_den = []
        for x in o:
            component_occupation_den.append(sum([pdf(gaussian_means[e],gaussian_variances[e],x)*component_prob[e] for e in range(m)]))
        for i in range(m):
            component_occupation_num = numpy.array([pdf(gaussian_means[i], gaussian_variances[i], x)*component_prob[i] for x in o])

            component_occupation = component_occupation_num/component_occupation_den
            tot_component_occupation = sum(component_occupation)
            gaussian_means[i] = sum([component_occupation[ind]*o[ind] for ind in range(len(o))])/tot_component_occupation
            gaussian_variances[i] = sum([component_occupation[ind]*(gaussian_means[i]-o[ind])**2 for ind in range(len(o))])/tot_component_occupation
            component_prob[i] = tot_component_occupation/len(o)
        if fixed_iterations:
            if counter >= fixed_iterations:
                break
        if sum([numpy.linalg.norm(gaussian_means[i], ord=2) for i,_ in enumerate(gaussian_means)]) == sum(
                [numpy.linalg.norm(previous_gaussian_means[i], ord=2) for i,_ in enumerate(previous_gaussian_means)]):
            break
        else:
            counter+=1
            print('GMM iteration {}'.format(str(counter)))
            previous_gaussian_means = gaussian_means.copy()
    return component_prob, gaussian_means, gaussian_variances

def classify(data, component_prob, gaussian_means, gaussian_variance):
    classified_data = []
    for point in data:
        assigned_class = 0
        for index, mean in enumerate(gaussian_means):
            if index == 0:
                prob = pdf(mean, gaussian_variance[0], point)*component_prob[0]
            else:
                new_prob = pdf(mean, gaussian_variance[index], point)*component_prob[index]
                if new_prob>prob:
                    prob = new_prob
                    assigned_class = index
        point = list(point)
        point.append(assigned_class)
        classified_data.append(point)
    return classified_data

def compute_mixture_pdf(obs,mixtures):
    prob = sum([mixtures[0][mix]*pdf(mixtures[1][mix], mixtures[2][mix], obs) for mix in range(len(mixtures[0]))])
    return prob

#
#
if __name__=='__main__':
    a = numpy.array(
        [[1, 4], [5, 2], [6, 7], [8, 9], [5, 10], [12, 15], [-2, -3], [5, 6], [23, 9], [14, 17], [19, 20], [13, 15]])

    component_prob, means, variances = GMM(a,5)

    classified = classify(a,component_prob,means,variances)

