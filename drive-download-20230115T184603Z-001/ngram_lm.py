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
    ngrams = []
    text = start_pad(c) + text
    for i in range(c, len(text)):
        context = text[i-c:i]
        char = text[i]
        ngrams.append((context, char))
    return ngrams


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

    def __init__(self, c, k=0):
        self.__context = c
        self.__k = k
        self.__vocab = set()
        self.__ngrams = {}

    def get_vocab(self) -> set:
        ''' Returns the set of characters in the vocab '''
        return self.__vocab

    def update(self, text: str):
        ''' Updates the model n-grams based on text
        ngram_list = ngrams(self.__context, text)
        for char in text:
            if char not in self.__vocab:
                self.__vocab.update(char)
        for n_gram in ngram_list:
            if n_gram in self.__ngrams:
                self.__ngrams[n_gram] += 1
            else:
                self.__ngrams[n_gram] = 1'''
        for char in text:
            self.__vocab.update(char)
        #print(text_list)
        ngrams_list = ngrams(self.__context, text)
        for ngram in ngrams_list:
            if ngram in self.__ngrams.keys():
                self.__ngrams[ngram] += 1
            else:
                self.__ngrams[ngram] = 1
        for ngram in self.__ngrams:
            self.__ngrams[ngram] += self.__k
        #print(self.__ngrams)

    def prob(self, context, char):
        ''' Returns the probability of char appearing after context '''
        ngram = context, char
        ngram_occurrences_start_with_context = self.__get_ngram_count_with_first_context(context)
        ngram_occurrences = 0
        if ngram_occurrences_start_with_context != 0:
            if ngram in self.__ngrams.keys():
                ngram_occurrences = self.__ngrams[ngram]
            return ngram_occurrences / ngram_occurrences_start_with_context
        return 1 / len(self.__vocab)

    def __get_ngram_count_with_first_context(self, context):
        '''get count of ngrams that start with char '''
        counter = 0
        for ngram, count in self.__ngrams.items():
            if ngram[0] == context:
                counter += count
        return counter

    def random_char(self, context):
        ''' Returns a random character based on the given context and the 
            n-grams learned by this model
            sorted_vocab = sorted(self.__vocab)
        random_num = random.random()
        probability = 0
        for char in sorted_vocab:
            if probability <= random_num:
                probability += self.prob(context, char)
            if probability > random_num:
                return char
            '''
        sorted_vocab = sorted(self.__vocab)
        random_num = random.random()
        probability = 0
        i = 0
        while probability < random_num and i < len(sorted_vocab):
            probability += self.prob(context, sorted_vocab[i])
            i += 1
        return sorted_vocab[i - 1]

    def random_text(self, length):
        ''' Returns text of the specified character length based on the
            n-grams learned by this model '''
        #probable_string = ""
        #starting_context = start_pad(self.__context)
        #while len(probable_string) < length:
            #if len(probable_string) == 0:
                #char = self.random_char(starting_context)
                #probable_string += char
            #else:
                #updated_context = probable_string[-self.__context]
                #char = self.random_char(updated_context)
                #probable_string += char
        #return probable_string
        probable_string = start_pad(self.__context)
        for i in range(length):
            probable_string += str(self.random_char(probable_string[len(probable_string) - self.__context:]))
        return probable_string



    def perplexity(self, text):
        ''' Returns the perplexity of text based on the n-grams learned by
            this model '''
        cross_entropy = 0
        for i in text:
            prob = self.prob(self.__context, text[i])
            cross_entropy -= math.log2(prob)

        perplexity = math.exp(cross_entropy / len(ngrams))
        return perplexity

    #iterates through the ngrams, searches ngrams[i][-1] to see if


################################################################################
# N-Gram Model with Interpolation
################################################################################

class NgramModelWithInterpolation(NgramModel):
    ''' An n-gram model with interpolation '''

    def __init__(self, c, k):
        self.__context = c
        self.__k = k
        self.__vocab = set()
        self.__ngrams = {}

    def set_k(self, k:int):
        self.__k = k

    def get_vocab(self):
        pass

    def update(self, text):
        ngram_list = ngrams(self.__context, text)
        for char in text:
            if char not in self.__vocab:
                self.__vocab.update(char)
        for n_gram in ngram_list:
            if n_gram in self.__ngrams:
                self.__ngrams[n_gram] += 1
            else:
                self.__ngrams[n_gram] = 1
        for n_gram in self.__ngrams:
            self.__ngrams[n_gram] += self.__k
        self.__vocab.add(self.__k)

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
    #m = NgramModel(1, 0)
    #m.update('abab')
    #print(m.get_vocab())
    #m.update('abcd')
    #print(m.get_vocab())
    #print(m.prob('a', 'b'))
    #print(m.prob('~', 'c'))
    #print(m.prob('b', 'c'))
    #random.seed(1)
    #print([m.random_char('') for i in range(25)])
    #print(m.random_text(25))
    n = create_ngram_model(NgramModel, 'shakespeare_input.txt', 3, 0)
    print(n.random_text(250))
    #print(m.perplexity('abcd'))