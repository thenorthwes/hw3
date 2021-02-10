from math import inf

import numpy as np

ENG_S_TRAIN = "./files/conll03_ner/eng.train.small"
ENG_S_DEV = "./files/conll03_ner/eng.dev.small"
ENG_S_TEST = "./files/conll03_ner/eng.NOTYET.small"
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

NER_SET = [O_TAG, B_MISC, I_MISC, B_PER, I_PER, B_ORG, I_ORG, B_LOC, I_LOC, B_ORG, I_ORG]
pos = {}
global_feature_counter = 1


def estimate_sentence(sentence, weight_vector):
    viterbi_table = []
    tag_stack = []
    features_list = []
    for i in range(len(sentence)):
        word_tuple = sentence[i]
        word = word_tuple[0]
        part_of_spch = word_tuple[1]
        viterbi_table.append([word, {}])  # Add a new col for each word Header is the word, dict for NERs
        words_feature_dict = {}

        # Words feature dict ends up with feature#, 1 for all features found about this word
        # POS
        words_feature_dict[pos[part_of_spch]] = 1
        # Print which feature is on and the bin val of on
        # print(words_feature_dict)
        # for feature in words_feature_dict.keys():
        #     print("Feature on {}, weight {}".format(feature, weight_vector[feature]))
        features_list.append(words_feature_dict)
        for ner in NER_SET:
            feature_pi = 0
            for feature in words_feature_dict.keys():
                # features are feature index, and then 1 {17, 1}, {12, 1} -- etc
                feature_pi = weight_vector[feature] * words_feature_dict[feature]
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
            viterbi_table[i][1][ner] = (final_pi, prev_tag)
        #print(viterbi_table[i])

    backindex = len(sentence)-1
    # walk backwards
    backpointer = max(viterbi_table[backindex][1].items(), key=lambda point: point[1][0])[0]
    tag_stack.append(backpointer)
    while True:  ## stop when we get to the first col with break
        #print(viterbi_table[backindex][0])
        backpointer = viterbi_table[backindex][1][backpointer][1]
        if backpointer is None:
            break
        tag_stack.append(backpointer)
        backindex -= 1
    tag_stack.reverse()

    # Write to output
    return tag_stack, features_list, {"Supposed to be the Φ(xi, yi)"}


def add_to_output(output, sentence, estimated_sequence):
    line = "{}\t{}\t{}\t{}\t{}"
    for i in range(len(sentence)):
        printLine = line.format(sentence[i][0], sentence[i][1], sentence[i][2], sentence[i][3], estimated_sequence[i])
        output.write(printLine)
        output.write("\n")
    output.write("\n")


def startPerceptronCycles(featureVector):
    output = open(OUTPUT, "w")
    training_sentences = []
    # make an array to work with
    with open(ENG_S_TRAIN, "r") as training_data:
        tagged_word = training_data.readline()
        sentence = []
        while tagged_word:
            word_tag_array = tagged_word.split()
            word_tag_tuple = (word_tag_array[0], word_tag_array[1], word_tag_array[2], word_tag_array[3])
            sentence.append(word_tag_tuple)
            tagged_word = training_data.readline()
            if tagged_word == "\n":
                training_sentences.append(sentence)
                sentence = []
                tagged_word = training_data.readline()
        print("End of file")

    extract_features(training_sentences)

    #  for every feature we have turned on -- make up a random weight to start with
    weight_vector = np.random.randint(-101, 101, global_feature_counter)
    weight_vector = weight_vector / 100
    weight_history = []
    # How many iterations
    for someloops in range(1):
        for sentence in training_sentences:
            viterbi_result = estimate_sentence(sentence, weight_vector)
            estimated_sequence = viterbi_result[0]
            correct_seq = np.array([(sentence[x][3]) for x in range(len(sentence))])
            print(correct_seq)
            print(estimated_sequence)
            # Test does our estimated Seq = the real seq?
            if np.array_equal(correct_seq, estimated_sequence):
                print("Yahoo")
            else:
                weight_history.append(weight_vector) # track history for average
                print("update weights")

    # do once when we are satisfied with our training
    for sentence in training_sentences:
        viterbi_result = estimate_sentence(sentence, weight_vector)
        estimated_sequence = viterbi_result[0]
        add_to_output(output, sentence, estimated_sequence)
    output.close()


def extract_features(training_sentences):
    # Turn on Features
    global global_feature_counter
    for sentence in training_sentences:
        for word_tuple in sentence:

            part_of_speech = word_tuple[1]
            if part_of_speech not in pos.keys():
                pos[part_of_speech] = global_feature_counter
                global_feature_counter = global_feature_counter + 1


def start():
    # Define feature vectors
    featureVector = ['pos: {}']
    # 'shape:{} {}'.format(shape, cur),
    # 'pos:{} {}'.format(pos, cur),
    # 'sc:{} {}'.format(sc, cur),
    # 'prefix:{} {}'.format(prefix, cur),
    # 'suffix:{} {}'.format(suffix, cur)]
    print(featureVector)
    # define weight vector and initiatlize
    startPerceptronCycles(featureVector)


# start of HW 3
if __name__ == '__main__':
    print("Starting HW 3")
    start()
