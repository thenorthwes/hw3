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


def estimate_sentence(sentence):
    tag_seq = []
    for word_tuple in sentence:
        # This is where i need to do featurey things and viterbi
        tag_seq.append(O_TAG)
    # Write to output
    return tag_seq


def add_to_output(output, sentence, estimated_sequence):
    line = "{}\t{}\t{}\t{}\t{}"
    for i in range(len(sentence)):
        printLine = line.format(sentence[i][0], sentence[i][1], sentence[i][2], sentence[i][3], estimated_sequence[i])
        output.write(printLine)
        output.write("\r\n")

    output.write("\r\n")


def startPerceptronCycles(featureVector):
    output = open(OUTPUT, "w")
    training_sentences = []
    # grab a sentence from training
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

    for sentence in training_sentences:
        estimated_sequence = estimate_sentence(sentence)
        add_to_output(output, sentence, estimated_sequence)
    output.close()


def start():
    # Define feature vectors
    featureVector = ['is_capped: {} {}',
                     'pos: {} {}']
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
