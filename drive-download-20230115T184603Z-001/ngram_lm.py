import math, random
'''
Hope Crisafi
Natural Language Processing
Project 1: N-gram Language Model
Last Updated: Jan 20, 2023
'''

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
        self.__c = c
        self.__k = k
        self.__vocab = set()
        self.__ngrams = {}

    def get_vocab(self):
        ''' Returns the set of characters in the vocab '''
        return self.__vocab

    def update(self, text):
        ''' Updates the model n-grams based on text'''
        for char in text:
            self.__vocab.update(char)
        ngrams_list = ngrams(self.__c, text)
        for ngram in ngrams_list:
            if ngram in self.__ngrams:
                self.__ngrams[ngram] += 1
            else:
                self.__ngrams[ngram] = 1

    def prob(self, context, char):
        ''' Returns the probability of char appearing after context '''
        ngram = context, char
        ngram_occurrences_start_with_context = self.__get_ngram_count_with_first_context(context)
        ngram_occurrences = 0
        if ngram_occurrences_start_with_context != 0:
            if ngram in self.__ngrams:
                ngram_occurrences = self.__ngrams[ngram]
            return (ngram_occurrences + self.__k) / (ngram_occurrences_start_with_context + self.__k * len(self.__vocab))
        return 1 / len(self.__vocab)

    def __get_ngram_count_with_first_context(self, context):
        '''get count of ngrams that start with char '''
        counter = 0
        for ngram, count in self.__ngrams.items():
            if ngram[0] == context:
                counter += count
        return counter

    def set_k(self, k):
        '''Set the value of k for smoothing'''
        self.__k = k

    def random_char(self, context):
        ''' Returns a random character based on the given context and the 
            n-grams learned by this model'''
        sorted_vocab = sorted(self.__vocab)
        random_float = random.random()
        probability = 0
        for char in sorted_vocab:
            if probability <= random_float:
                probability += self.prob(context, char)
            if probability > random_float:
                return char

    def random_text(self, length):
        ''' Returns text of the specified character length based on the
            n-grams learned by this model '''
        probable_string = start_pad(self.__c)
        for i in range(length):
            updated_context = probable_string[-self.__c:]
            probable_string += self.random_char(updated_context)
        return probable_string[self.__c:]

    def perplexity(self, text):
        ''' Returns the perplexity of text based on the n-grams learned by
            this model '''
        n_grams = ngrams(self.__c, text)
        total_probability = 0
        for ngram in n_grams:
            prob = self.prob(ngram[0], ngram[1])
            if prob == 0:
                return float('inf')
            total_probability += math.log(prob)
        perplexity = math.exp(-(1 / len(text)) * total_probability)
        return perplexity



################################################################################
# N-Gram Model with Interpolation
################################################################################

class NgramModelWithInterpolation(NgramModel):
    ''' An n-gram model with interpolation '''

    def __init__(self, c, k):
        super().__init__(c, k)
        self.__lambdas = []
        for i in range(c+1):
            self.__lambdas.append(1/(c+1))

    def set_lambda(self, lam_vals):
        ''' sets the lambda values '''
        self.__lambda = lam_vals

    def update(self, text):
        ''' Updates the model n-grams based on text '''
        all_order_ngrams = []

        for char in text:
            if char not in super().get_vocab():
                self._NgramModel__vocab.update(char)

        for i in range(self._NgramModel__c+1):
            n_grams = ngrams(i, text)
            all_order_ngrams.append(n_grams)

        for n_grams in all_order_ngrams:
            for n_gram in n_grams:
                if n_gram in self._NgramModel__ngrams:
                    self._NgramModel__ngrams[n_gram] += 1
                else:
                    self._NgramModel__ngrams[n_gram] = 1

    def prob(self, context, char):
        ''' Returns the probability of char appearing after context '''
        probability = 0
        for i in range(self._NgramModel__c+1):
            prob1 = super().prob(context[len(context) - i:], char)
            prob = prob1 * self.__lambdas[i]
            probability += prob
        return probability

################################################################################
# Your N-Gram Model Experimentations
################################################################################
def NgramModel_tests():
    ''' NgramModel class tests '''
    print("----Runnin NgramModel tests----")
    m = NgramModel(1, 0)
    m.update('abab')
    print(m.get_vocab(), '(expected: {’b’, ’a’})')
    m.update('abcd')
    print(m.get_vocab(), '(expected: {’b’, ’a’, ’c’, ’d’}')
    print('WITHOUT INTERP')
    print(m.prob('a', 'a'), '(expected: 1.0)')
    print(m.prob('a', 'b'), '(expected: 1.0)')
    #m.set_k(.15)
    print(m.prob('~', 'c'), '(expected: 0.0)')
    print(m.prob('b', 'c'), '(expected: 0.5)')
    random.seed(1)
    print([m.random_char('') for i in range(25)])
    print(m.random_text(25))
    shakespeare = create_ngram_model(NgramModel, 'shakespeare_input.txt', 4)
    sonnets = create_ngram_model(NgramModel, 'shakespeare_sonnets.txt', 4)
    nyt = create_ngram_model(NgramModel, 'nytimes_article.txt', 4)
    print("shakes perp: ",n.perplexity('shakespeare_input.txt'))
    print("sonnets perp: ", shakespeare.perplexity('shakespeare_sonnets.txt')) #shakes to sonnets
    print("nyt perp: ", n.perplexity('nytimes_article.txt')) # shakes to nyt
    print(m.perplexity('abcd'), '(expected: 1.189207115002721)')
    #print(n.random_text(250))
    print(m.perplexity('abcda'), '(expected: 1.515716566510398)')
    print(m.perplexity('the quick brown fox jumped over the lazy sleeping dog'))

def NgramModelWithInterpolation_tests():
    ''' NgramModelWithInterpolation class tests '''
    print("----Runnin NgramModelWithInterpolation tests----")
    m = NgramModelWithInterpolation(1, 0)
    m.update('abab')
    print('WITH INTERP')
    m.set_lambda(.33)
    print(m.prob('a','a'), '(expected: 0.25)')
    print(m.prob('a', 'b'), '(expected: 0.75)')
    a = NgramModelWithInterpolation(2, 1)
    a.update('abab')
    a.update('abcd')
    print(a.prob('~a', 'b'), '(expected: 0.4682539682539682)')
    print(a.prob('~c', 'd'), '(expected: 0.27222222222222225)')

def CitiesNgrams_tests():
    #COUNTRY_CODES = ['af', 'cn', 'de', 'fi', 'fr', 'in', 'ir', 'pk', 'za']
    af = create_ngram_model(NgramModel, 'af.txt', 3)
    cn = create_ngram_model(NgramModel, 'cn.txt', 3)
    de = create_ngram_model(NgramModel, 'de.txt', 3)
    fi = create_ngram_model(NgramModel, 'fi.txt', 3)
    fr = create_ngram_model(NgramModel, 'fr.txt', 3)
    ind = create_ngram_model(NgramModel, 'in.txt', 3)
    ir = create_ngram_model(NgramModel, 'ir.txt', 3)
    pk = create_ngram_model(NgramModel, 'pk.txt', 3)
    za = create_ngram_model(NgramModel, 'za.txt', 3)

    country_language_models = [af, cn, de, fi, fr, ind, ir, pk, za]

    for language in country_language_models:
        pass


if __name__ == "__main__":
    NgramModel_tests()
    NgramModelWithInterpolation_tests()

