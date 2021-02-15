import time
from copy import deepcopy
from math import inf

import numpy as np

ENG_S_TRAIN = "./files/conll03_ner/eng.train.small"
ENG_S_DEV = "./files/conll03_ner/eng.dev.small"
ENG_S_TEST = "./files/conll03_ner/eng.test.small"
OUTPUT = "./files/out/output.txt"

O_TAG = "O"
B_MISC = "B-MISC"
I_MISC = "I-MISC"
B_PER = "B-PER"
I_PER = "I-PER"
B_LOC = "B-LOC"
I_LOC = "I-LOC"
B_ORG = "B-ORG"
I_ORG = "I-ORG"

NER_SET = [O_TAG, B_MISC, I_MISC, B_PER, I_PER, B_LOC, I_LOC, B_ORG, I_ORG]
pos = {}
sc = {}
wd = {}
st = {}
en = {}
cap = {}

global_feature_counter = 1

POS_FEATURE_ON = True
SC_FEATURE_ON = True
WORD_NRE_ON = True
START_NRE_ON = True
END_NRE_ON = True
CAP_NRE_ON = True

PRINT_INTERVALS = False


# Viterbi algorithm for estimating sequence
# Input is a sentence tuple array
# weight vector is the current weight vector
# Single table is used and each cell is a tuple that tracks all the info, what features were on, what previous NER
# Tag got us to this cell etc... this way we can can reconstruct the feature set and tags that got us to the end
def estimate_sentence(sentence, weight_vector):
    viterbi_table = []
    tag_stack = []
    features_list_viterbi = []
    features_list_gold = []

    # for every word
    for i in range(len(sentence)):
        word_tuple = sentence[i]
        word = word_tuple[0]
        part_of_spch = word_tuple[1]
        syntatic_chunk = word_tuple[2]
        correct_ner = word_tuple[3]
        viterbi_table.append([word, {}])  # Add a new col for each word Header is the word, dict for NERs

        # For every NER Tag
        for ner in NER_SET:
            # Words feature dict ends up with feature#, 1 for all features found about this word
            words_feature_dict = get_features_for_word(ner, word, part_of_spch, syntatic_chunk, i == 0,
                                                       i + 1 == len(sentence))

            if correct_ner == ner:
                features_list_gold.append(words_feature_dict)

            # the features for this NER selected with this viterbi is pushed into the table

            feature_pi = 0
            for feature in words_feature_dict.keys():
                # features are feature index, and then 1 {17, 1}, {12, 1} -- etc
                feature_pi = (weight_vector[feature] * words_feature_dict[feature]) + feature_pi
            # For every NER -- fill our pi for the NER tag
            final_pi = 0
            prev_tag = None
            if i == 0:  # Start of sentence, just fill out
                final_pi = feature_pi
            else:  # find best path to this node
                max_pi = -inf
                prev_node = None
                for prev_ner in NER_SET:
                    # get prob for this prev_ner
                    prev_node_pi = viterbi_table[i - 1][1][prev_ner][0]
                    if (prev_node_pi + feature_pi) > max_pi:
                        max_pi = prev_node_pi + feature_pi
                        prev_node = prev_ner
                final_pi = max_pi
                prev_tag = prev_node
            viterbi_table[i][1][ner] = (final_pi, prev_tag, words_feature_dict)

    backindex = len(sentence) - 1
    # walk backwards
    cell = max(viterbi_table[backindex][1].items(), key=lambda point: point[1][0])
    backpointer = cell[0]
    tag_stack.append(backpointer)
    # Append the features that gave us this cell
    features_list_viterbi.append(cell[1][2])
    while True:  # stop when we get to the first col with break
        # print(viterbi_table[backindex][0])
        backpointer = viterbi_table[backindex][1][backpointer][1]
        if backpointer is None:
            break
        features_list_viterbi.append(viterbi_table[backindex][1][backpointer][2])
        tag_stack.append(backpointer)
        backindex -= 1
    tag_stack.reverse()
    features_list_viterbi.reverse()  # doesnt matter but yah know whatever

    return tag_stack, features_list_viterbi, features_list_gold


# Words feature dict ends up with feature#, 1 for all features found about this word
#  that feature number is the index to be used in the weight array
# For now, really just using 1s for freature sets...
def get_features_for_word(ner, word, part_of_spch, syntatic_chnk, start_of_sent, end_of_sent) -> dict:
    words_feature_dict = {}
    if WORD_NRE_ON and (word+ner) in wd.keys():
        words_feature_dict[wd[word + ner]] = 1
    if POS_FEATURE_ON and (part_of_spch + ner) in pos.keys():
        words_feature_dict[pos[part_of_spch + ner]] = 1
    if SC_FEATURE_ON and (syntatic_chnk+ner) in sc.keys():
        words_feature_dict[sc[syntatic_chnk + ner]] = 1
    if START_NRE_ON:
        if start_of_sent:
            words_feature_dict[st["ST_TRUE" + ner]] = 1
        else:
            words_feature_dict[st["ST_FALSE" + ner]] = 1
    if END_NRE_ON:
        if end_of_sent:
            words_feature_dict[en["END_TRUE" + ner]] = 1
        else:
            words_feature_dict[en["END_FALSE" + ner]] = 1
    if CAP_NRE_ON:
        if word[0].isupper():
            words_feature_dict[cap["CAP_TRUE" + ner]] = 1
        else:
            words_feature_dict[cap["CAP_FALSE" + ner]] = 1
    return words_feature_dict

# Write to out put for perl script
def add_to_output(output, sentence, estimated_sequence):
    line = "{}\t{}\t{}\t{}\t{}"
    for i in range(len(sentence)):
        printLine = line.format(sentence[i][0], sentence[i][1], sentence[i][2], sentence[i][3], estimated_sequence[i])
        output.write(printLine)
        output.write("\n")
    output.write("\n")


def startPerceptronCycles():
    # make an array of sentences to work with
    # training file is merged into an array of arrays.
    # top level array -- each item is a sentence
    # That sentence is an array of tuples
    # Each tuple is a a tagged word (WORD, PART OF SPEECH, SYNTACTIC CHUNK, NER TAG)
    training_sentences = get_sentences_as_array(ENG_S_TRAIN)

    # extract features from our templates
    # Boolean flags are available to turn on / off
    extract_features(training_sentences)

    #  for every feature we have turned on -- make up a random weight to start with
    weight_vector = np.random.randint(-101, 101, global_feature_counter)
    weight_vector = weight_vector
    weight_avg = deepcopy(weight_vector)
    weight_update_counter = 1
    # How many iterations of perceptron should we run
    # after experimental evidence, 150~ seemed to achieve stability
    for someloops in range(1, 151):
        if someloops % 5 == 0:  # Print out evidence of progress so we don't bored
            end = "\t\t{}\n".format(time.process_time(), 2)
            if PRINT_INTERVALS:  # This flag will print out incremental results to produce learning curve
                output = open(OUTPUT + str(someloops), "w")
                for sentence in training_sentences:
                    viterbi_result = estimate_sentence(sentence, weight_vector)
                    estimated_sequence = viterbi_result[0]
                    add_to_output(output, sentence, estimated_sequence)
                output.close()
        else:
            end = ""
        print(".", end=end)

        # For every sentence in the file
        for sentence in training_sentences:
            # Use viterbi to estimate NER sequence
            # Result is a tuple with the estimated Sequence, the estimated features, and the gold features
            viterbi_result = estimate_sentence(sentence, weight_vector)
            estimated_sequence = viterbi_result[0]

            # Fetch the correct sequence off the sentence
            correct_seq = np.array([(sentence[x][3]) for x in range(len(sentence))])

            # Test does our estimated Seq = the real seq?
            if np.array_equal(correct_seq, estimated_sequence):
                nop = 1  # For readability -- No operation
            else:  # Didnt match? Then do update
                for v_vector_list in viterbi_result[1]:  # Weight down for bad
                    for v in v_vector_list:
                        weight_vector[v] -= v_vector_list[v]
                for g_vector_list in viterbi_result[2]:  # Weight up for good ones
                    for g in g_vector_list:
                        weight_vector[g] += g_vector_list[g]
                # Increment our denominator for tracking
                weight_update_counter += 1
                weight_avg = np.add(weight_avg, weight_vector)

    # Avg Weights to prevent overfitting
    weight_avg = np.divide(weight_avg, weight_update_counter)
    # Evaluate on Dev
    dev_sentences = get_sentences_as_array(ENG_S_DEV)
    #
    outputDev = open(OUTPUT+"dev", "w")
    for sentence in dev_sentences:
        viterbi_result = estimate_sentence(sentence, weight_avg)
        estimated_sequence = viterbi_result[0]
        add_to_output(outputDev, sentence, estimated_sequence)
    outputDev.close()

    # Evaluate on Dev
    test_sentences = get_sentences_as_array(ENG_S_TEST)
    #
    outputTest = open(OUTPUT + "test", "w")
    for sentence in test_sentences:
        viterbi_result = estimate_sentence(sentence, weight_avg)
        estimated_sequence = viterbi_result[0]
        add_to_output(outputTest, sentence, estimated_sequence)
    outputTest.close()

    # do once when we are satisfied with our training
    outputFinal = open(OUTPUT, "w")
    for sentence in training_sentences:
        viterbi_result = estimate_sentence(sentence, weight_vector)
        estimated_sequence = viterbi_result[0]
        add_to_output(outputFinal, sentence, estimated_sequence)
    outputFinal.close()

# Get all the sentences from a file as an array of sentence arrays where each item is a tuple
# [ ] [ ] [ ] [ a ]   <-- Sentences
# [ a ] = [(word, pos, sc, NER), (etc)]
def get_sentences_as_array(path):
    sent = []
    with open(path, "r") as data:
        tagged_word = data.readline()
        sentence = []
        while tagged_word:
            word_tag_array = tagged_word.split()
            word_tag_tuple = (word_tag_array[0], word_tag_array[1], word_tag_array[2], word_tag_array[3])
            sentence.append(word_tag_tuple)
            tagged_word = data.readline()
            if tagged_word == "\n":
                sent.append(sentence)
                sentence = []
                tagged_word = data.readline()
    return sent

# All features are contextualized by an NER tag -- global feature value is tracked
# that global feature number is the index in the weight vectory
# When we get features of a word -- we turn them on by turning on the index
def extract_features(training_sentences):
    # Turn on Features
    global global_feature_counter
    partsOfSpeech = {}
    syntaticChunk = {}
    words = {}
    for sentence in training_sentences:
        for word_tuple in sentence:
            part_of_speech = word_tuple[1]
            partsOfSpeech[part_of_speech] = 1
            sychunk = word_tuple[2]
            syntaticChunk[sychunk] = 1
            word = word_tuple[0]
            words[word] = 1

    # Contextualize features by NERs
    for ner in NER_SET:
        if POS_FEATURE_ON:
            for p in partsOfSpeech:
                # Results in something like NNPB-ORG | feature_index
                pos[p + ner] = global_feature_counter
                global_feature_counter += 1
        if SC_FEATURE_ON:
            for s in syntaticChunk:
                # Results in something like I-NPB-ORG
                sc[s + ner] = global_feature_counter
                global_feature_counter += 1
        if WORD_NRE_ON:
            for w in words:
                # Results in something like WORDB-ORG
                wd[w + ner] = global_feature_counter
                global_feature_counter += 1
        if START_NRE_ON:
            st["ST_TRUE" + ner] = global_feature_counter
            global_feature_counter += 1
            st["ST_FALSE" + ner] = global_feature_counter
            global_feature_counter += 1
        if END_NRE_ON:
            en["END_TRUE" + ner] = global_feature_counter
            global_feature_counter += 1
            en["END_FALSE" + ner] = global_feature_counter
            global_feature_counter += 1
        if CAP_NRE_ON:
            cap["CAP_TRUE" + ner] = global_feature_counter
            global_feature_counter += 1
            cap["CAP_FALSE" + ner] = global_feature_counter
            global_feature_counter += 1


def start():
    print("HW 3")
    # This is really where the whole app takes place
    startPerceptronCycles()
    print("HW3 Complete in: {} sec.".format(time.process_time(),3))


# start of HW 3
if __name__ == '__main__':
    print("Starting HW 3")
    start()
