import numpy as np
import pandas as pd
import math
import GaussianCalculator



def uniform_start(mfccs, num_states):
    # Segment the data uniformly and compute means and standard deviation from the thus obtained clusters
    Observation_probabilities = {digit: {'{}_{}'.format(digit, i):[] for i in range(num_states)} for digit in
                   mfccs.keys()}
    for digit in mfccs:
        for frames in mfccs[digit]:
            # for frames in recordings:
                modulo = len(frames)//num_states
                index = 0
                for state in Observation_probabilities[digit]:
                    try:
                        Observation_probabilities[digit][state].extend([x for x in frames[index:index+modulo]])
                        index+=modulo
                    except IndexError:
                        Observation_probabilities[digit][state].extend([x for x in frames[index:]])
    parameters={}
    for digit in mfccs:
        parameters[digit]={}
        for state in Observation_probabilities[digit]:
            means = GaussianCalculator.m(Observation_probabilities[digit][state])
            parameters[digit][state] = [means, GaussianCalculator.s(means,Observation_probabilities[digit][state])]

    return Observation_probabilities, parameters




def Create_Transition_tables(num_hidden_states):
    # Create an initial transitions table
    digits = set(['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine'])

    HMMS_labels = {digit: ['start'] + ['{}_{}'.format(digit, i) for i in range(num_hidden_states)] + ['end'] for digit in digits}

    Transitions = {}

    for digit in HMMS_labels:
        d = {"start": [0] + [1] + [0 for i in range(num_hidden_states)]}
        col_names = ['start']
        for index, value in enumerate(HMMS_labels[digit][1:-1]):
            d[value] = [0 for i in range(num_hidden_states + 2)]
            d[value][index + 1] = d[value][index + 2] = 0.5
            col_names.append(value)
        d['end'] = [0 for i in range(num_hidden_states + 2)]
        col_names.append('end')
        Transitions[digit] = pd.DataFrame(d).T
        Transitions[digit].columns = col_names
    return Transitions

class MyViterbiDecoder:
    NLL_ZERO = 1e10  # define a constant representing -log(0).  This is really infinite, but approximate

    # it here with a very large number

    def __init__(self, mfcc_frames, observation_probabilities,transition_probabilities, num_hidden_states=3):
        """Set up the decoder class with an audio file and WFST f
        """
        self.forward_counter = 0

        self.observations = mfcc_frames

        self.digits = ['zero', 'one', 'two','three','four','five', 'six','seven','eight','nine']

        self.num_states = num_hidden_states

        self.observation_probabilities = observation_probabilities

        self.HMMS_labels = {digit: ['start']+['{}_{}'.format(digit, i) for i in range(self.num_states)]+['end'] for digit in self.digits}

        self.Transitions = transition_probabilities

        self.initialise_decoding()


    def initialise_decoding(self):
        """set up the values for V_j(0) (as negative log-likelihoods)

        """

        self.V = {}  # stores likelihood along best path reaching state j
        self.B = {}  # stores identity of best previous state reaching state j
        for digit in self.digits:
            self.V[digit] = []
            self.B[digit] = []
            for t in range(len(self.observations)+1):
                self.V[digit].append([self.NLL_ZERO] * (self.num_states+2))
                self.B[digit].append([-1] * (self.num_states+2))
                self.V[digit][0][0] = 0.0

    def forward_step(self, t):

        # init best_cost, which will be used in beam search. The cost larger than `best_cost` + `beam_width` (for pruning).
        best_cost = self.NLL_ZERO

        for digit in self.digits:

            for i,v in enumerate(self.HMMS_labels[digit][:-1]):

                if not self.V[digit][t - 1][i] == self.NLL_ZERO:  # no point in propagating states with zero probability

                  for arc in [i,i+1]:
                    """Given the left-to-right topology of this HMMs, just the current (self-loop) and next state
                    transitions need to be evaluated"""

                    try:
                        tp = -math.log(self.Transitions[digit].lookup([self.HMMS_labels[digit][i]],
                                                                      [self.HMMS_labels[digit][arc]]))  # transition prob
                    except ValueError:
                        continue # this happen when there is no valid transition between states
                    try:
                        if len(self.observation_probabilities[digit][self.HMMS_labels[digit][arc]])==2:
                            m,s = self.observation_probabilities[digit][self.HMMS_labels[digit][arc]]
                            ep = -math.log(GaussianCalculator.pdf(m,s,self.observations[t-1]))
                        else:
                            ep = -math.log(GaussianCalculator.compute_mixture_pdf(self.observations[t-1],
                                                                                      self.observation_probabilities[digit][self.HMMS_labels[digit][arc]]))  # emission negative log prob
                    except KeyError:
                        continue # this happen when the observation probability is evaluated for the end state
                    self.forward_counter += 1  # increase the forward counter by 1
                    prob = tp + ep + self.V[digit][t - 1][i]  # they're logs

                    # if the nega logprob is larger than the lowest at this time step plus the beam width, does nothing.
                    if prob > best_cost + self.beam_width:
                        continue

                    # else if it is lower than the current viterbi value at time t state j, update the viterbi value and write the backpointer...
                    elif prob < self.V[digit][t][arc]:
                        # Below conditions apply the cost of being in a final state (if any) to penalise words insertion
                        self.V[digit][t][arc] = prob
                        self.B[digit][t][arc] = i

                        # update the BEST_COST
                        best_cost = prob

    def finalise_decoding(self):
        """ this incorporates the probability of terminating at each state
        """
        self.finished = []

        for digit in self.digits:
            tp = tp = -math.log(self.Transitions[digit].lookup([self.HMMS_labels[digit][-2]],
                                                                      [self.HMMS_labels[digit][-1]]))
            self.finished.append(self.V[digit][-1][-2]+tp)

        self.result = self.digits[np.argmin(self.finished)]
        print(self.result)


    def decode(self, beam_width=10000):
        self.initialise_decoding()
        t = 1
        # add instance variable: beam_width
        self.beam_width = beam_width
        while t <= len(self.observations):
            self.forward_step(t)
            # self.traverse_epsilon_arcs(t)
            t += 1
        self.finalise_decoding()

    def backtrace(self):

      best_state_sequence = {}
      for digit in self.digits:


        best_final_state = self.num_states  # argmin
        best_state_sequence[digit] = [best_final_state]

        t = len(self.observations) # ie T
        j = best_final_state
        prev_j = -1
        while t >= 0:
            i = self.B[digit][t][j]
            best_state_sequence[digit].append(i)
            j = i
            t -= 1

        best_state_sequence[digit].reverse()

      return best_state_sequence

def viterbi_training(digits, mfccs, parameters, Transitions, num_hidden_states, mixtures=False):
    '''
    :param digits: the dictionary including the digit names as values and the digit symbols as keys
    :param mfccs: the parameterasied acoustic observations, consisting of 12 MFCC + energy for each frame
    :param parameters: the initial means and variances for the model, obtained via uniformly segmented the observations
    :param Transitions: the initial transition probabilities between states, consisting in 0.5 self-loop and 0.5 next transition
    :param num_hidden_states: the number of emitting hidden state for the model
    :param mixtures: if true, it includes the estimation of mixture parameters in the process
    :return:
    '''
    accumulator = {d:{d1:[] for d1 in parameters[d].keys()} for d in digits.values()}
    # Decoding the data with current parameters
    for digit in digits.values():
        best_transitions = []
        best_sequences = []
        for obs in mfccs[digit]:
            vit = MyViterbiDecoder(obs,parameters, Transitions, num_hidden_states)
            vit.digits = [digit]
            vit.decode()
            best_sequence = vit.backtrace()
            best_sequences.extend(best_sequence[digit])
            best_transitions.extend([str(best_sequence[digit][index+1])+'-'+str(el) for index,el
                                     in enumerate(best_sequence[digit][2:])])
            for index, seq in enumerate(best_sequence[digit][1:]):
                if index==0:
                    continue
                # Assigning to the newly segmented data the current observation, according to the state it belonged from the decoding process
                accumulator[digit][vit.HMMS_labels[digit][seq]].append(obs[index-1])
            accumulator[digit][vit.HMMS_labels[digit][3]].append(obs[-1])
        for col in vit.HMMS_labels[digit][1:-1]:
            # estimating new transitions as an average of the transition identities
            num = best_transitions.count(str(vit.HMMS_labels[digit].index(col)) + '-' + str(vit.HMMS_labels[digit].index(col)))
            den = best_sequences.count(vit.HMMS_labels[digit].index(col))
            Transitions[digit][col][col] = num/den
            Transitions[digit][vit.HMMS_labels[digit][int(col.split('_')[1])+2]][col] = 1-Transitions[digit][col][col]

    new_parameters = {}
    if not mixtures:
        for digit in mfccs:
            new_parameters[digit] = {}
            for state in accumulator[digit]:
                means = GaussianCalculator.m(accumulator[digit][state])
                new_parameters[digit][state] = [means, GaussianCalculator.s(means, accumulator[digit][state])]
    else:
        for digit in mfccs:
            new_parameters[digit] = {}
            for state in accumulator[digit]:
                components_prob, means, variances = GaussianCalculator.GMM(accumulator[digit][state],mixtures,fixed_iterations=10)
                new_parameters[digit][state] = [components_prob, means, variances]
    return new_parameters, Transitions, accumulator


def calculate_accuracy(digits, mfccs, parameters, Transitions, num_hidden_states):
    tot = 0
    corr = 0
    for digit in digits.values():
        for obs in mfccs[digit]:
            tot+=1
            vit = MyViterbiDecoder(obs,parameters, Transitions, num_hidden_states)
            vit.decode()
            if vit.result==digit:
                corr+=1
    accuracy = corr/tot
    print(accuracy)
    return accuracy

def increase_number_mixture(digits,obs,n_mix=3):
    new_parameters = {}
    for digit in digits.values():
        new_parameters[digit] = {}
        for state in obs[digit]:
            print('Increasing number of mixture components to {} for state {} of digit {}'.format(str(n_mix), state[-1],
                                                                                                  digit))
            new_parameters[digit][state] = GaussianCalculator.GMM(obs[digit][state], n_mix)
    return new_parameters