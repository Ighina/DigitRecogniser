#!/usr/bin/env python

import os
import argparse
import scipy.io.wavfile as wav
from python_speech_features import mfcc
from HMMs import *
import pickle

parser = argparse.ArgumentParser(
    description="A very, very basic digit recogniser")

parser.add_argument('--train_dir',
                    default="./Train",
                    help='Directory for training data')
parser.add_argument('--test_dir',
                    default="./Test",
                    help='Directory for test data')
parser.add_argument('--number_of_states', '-states', dest='s', default=3, type=int,
                    help='Number of states for each HMM')
parser.add_argument('--mixture_components', '-mix', default=0, type=int, help='Number of mixture components to use, if not included\
                                                                      default is 0 and GMMs wont be used')
parser.add_argument('--output', default='saved_dictionaries', type=str,
                    help='Where to save the dictionaries containing the trained parameters (the file names are fixed and\
                    consist in viterbi_parameters.pkl for the viterbi trained model, transitions.pkl for the transitions\
                    and mixture_{numofmixtures}.pkl for the GMM model)')

args = parser.parse_args()

digits = {'0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four', '5': 'five',
          '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine'}

mfccs = {'zero': [], 'one': [], 'two': [], 'three': [], 'four': [], 'five': [],
         'six': [], 'seven': [], 'eight': [], 'nine': []}

mfccs_test = {'zero': [], 'one': [], 'two': [], 'three': [], 'four': [], 'five': [],
              'six': [], 'seven': [], 'eight': [], 'nine': []}

for root, dirs, files in os.walk(args.train_dir, topdown=False):
    for file in files:
        digit = digits[file[0]]
        rate, signal = wav.read(os.path.join(args.train_dir, file))
        mfccs[digits[file[0]]].append(mfcc(signal, rate, appendEnergy=True))

for root, dirs, files in os.walk(args.test_dir, topdown=False):
    for file in files:
        digit = digits[file[0]]
        rate, signal = wav.read(os.path.join(args.test_dir, file))
        mfccs_test[digits[file[0]]].append(mfcc(signal, rate, appendEnergy=True))

states = args.s

segments, parameters = uniform_start(mfccs, states)

Transitions = Create_Transition_tables(states)

acc = calculate_accuracy(digits, mfccs_test, parameters, Transitions, states)

print('Initial accuracy with uniformly segmented data: {}'.format(str(acc)))

print('Initialising Viterbi training...')

new_parameters, new_Transitions, accumulator = viterbi_training(digits, mfccs, parameters, Transitions, states,
                                                   mixtures=False)

acc = calculate_accuracy(digits, mfccs_test, new_parameters, new_Transitions, states)

print('Accuracy after Viterbi training: {}'.format(str(acc)))

with open(args.output + '/viterbi_parameters.pkl', 'wb') as f:
    pickle.dump(new_parameters, f)

with open(args.output + '/transitions.pkl', 'wb') as f:
    pickle.dump(new_Transitions, f)

if args.mixture_components:
    print('Increasing gaussian mixtures to {}'.format(str(args.mixture_components)))

    mixture_parameters = increase_number_mixture(digits, accumulator, n_mix=args.mixture_components)

    acc = calculate_accuracy(digits, mfccs_test, mixture_parameters, new_Transitions, states)

    print('Accuracy after increasing mixture components: {}'.format(str(acc)))

    with open(args.output + '/mixture_{}.pkl'.format(str(args.mixture_components)), 'wb') as f:
        pickle.dump(mixture_parameters, f)
