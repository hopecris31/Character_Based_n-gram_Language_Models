import math, random

# Hope Crisafi
# Natural Language Procesing Project 1
# n-gram models

################################################################################
# Utility Functions
################################################################################

COUNTRY_CODES = ['af', 'cn', 'de', 'fi', 'fr', 'in', 'ir', 'pk', 'za']


def start_pad(c):
    ''' Returns a padding string of length c to append to the front of text
        as a pre-processing step to building n-grams. c = n-1 '''
    return '~' * c


def ngrams(c, text):
    ''' Returns the ngrams of the text as tuples where the first element is
        the length-c context and the second is the character '''
    # c is the number of characters to consider for the context (ex 2 is trigram)
    # start index is c, the amount of characters we are going to consider in the model
    # tuple contains the words that cone before word to find, and word to find, each char separated by a comma
    # returns a list of tuples
    pass


def create_ngram_model(model_class, path, c=2, k=0):
    ''' Creates and returns a new n-gram model trained on the entire text
        found in the path file '''
    model = model_class(c, k)
    with open(path, encoding='utf-8', errors='ignore') as f:
        model.update(f.read())
    return model


def create_ngram_model_lines(model_class, path, c=2, k=0):
    '''Creates and returns a new n-gram model trained line by line on the
        text found in the path file. '''
    model = model_class(c, k)
    with open(path, encoding='utf-8', errors='ignore') as f:
        for line in f:
            model.update(line.strip())
    return model


################################################################################
# Basic N-Gram Model
################################################################################

class NgramModel(object):
    ''' A basic n-gram model using add-k smoothing '''

    def __init__(self, c, k):
        """
        :param c:
        :param k:
        """
        self.__context = c
        self.__k = k
        self.__vocab = set()
        self.__ngrams = {}

    def get_vocab(self) -> set:
        ''' Returns the set of characters in the vocab '''
        return self.__vocab

    def update(self, text: str):
        ''' Updates the model n-grams based on text '''
        text_list = []
        for i in range(self.__context):
            text_list.append('~')
        for char in range(len(text)):
            self.__vocab.update(text[char])
            text_list.append(text[char])
        # print(text_list)
        end = self.__context + 1
        for i in range(len(text_list) - self.__context):
            n_gram_list = text_list[i:end]
            n_gram = ''.join(n_gram_list)
            # print(n_gram)
            if n_gram in self.__ngrams:
                self.__ngrams[n_gram] += 1
            else:
                self.__ngrams.update({n_gram: 1})
            end += 1

    def prob(self, context: str, char: str):
        ''' Returns the probability of char appearing after context '''
        ngram = context + char
        if context and char in self.__vocab:
            ngram_occurrences = self.__ngrams.get(ngram)
            ngram_occurrences_start_with_context = self.__get_ngram_count_with_first_context(context)
            try:
                return ngram_occurrences / ngram_occurrences_start_with_context
            except Exception:
                return 0.0
        else:
            # print(len(self.__vocab))
            return 1 / len(self.__vocab)

    def __get_ngram_count_with_first_context(self, context: str):
        '''get count of ngrams that start with char '''
        counter = 0
        for ngram, count in self.__ngrams.items():
            if ngram[0] == context:
                counter += count

        return counter

    def random_char(self, context):
        ''' Returns a random character based on the given context and the 
            n-grams learned by this model '''
        sorted_vocab = sorted(self.__vocab)
        random_num = random.random()
        probability = 0
        i = 0
        while probability <= random_num and i < len(sorted_vocab):
            probability += self.prob(context, sorted_vocab[i])
            i += 1
        return sorted_vocab[i - 1]

    def random_text(self, length):
        ''' Returns text of the specified character length based on the
            n-grams learned by this model '''


    def perplexity(self, text):
        ''' Returns the perplexity of text based on the n-grams learned by
            this model '''
        pass


################################################################################
# N-Gram Model with Interpolation
################################################################################

class NgramModelWithInterpolation(NgramModel):
    ''' An n-gram model with interpolation '''

    def __init__(self, c, k):
        pass

    def get_vocab(self):
        pass

    def update(self, text):
        pass

    def prob(self, context, char):
        pass


################################################################################
# Your N-Gram Model Experimentations
################################################################################

# Add all code you need for testing your language model as you are
# developing it as well as your code for running your experiments
# here.
#
# Hint: it may be useful to encapsulate it into multiple functions so
# that you can easily run any test or experiment at any time.


if __name__ == "__main__":

    m = NgramModel(1, 0)
    m.update('abab')
    print(m.get_vocab())
    m.update('abcd')
    print(m.get_vocab())
    print(m.prob('a', 'b'))
    print(m.prob('~', 'c'))
    print(m.prob('b', 'c'))
    print('_____')

    n = NgramModel(0, 0)
    n.update('abab')
    n.update('abcd')
    random.seed(1)
    print([n.random_char('') for i in range(25)])
    # print(m.random_text(25))
    # n = create_ngram_model(NgramModel, 'shakespeare_input.txt', 6, 0)
    # print(n.random_text(250))
