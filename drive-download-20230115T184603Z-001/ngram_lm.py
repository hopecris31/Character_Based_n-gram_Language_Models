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
        context = text[i - c:i]
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
            return (ngram_occurrences + self.__k) / (
                        ngram_occurrences_start_with_context + self.__k * len(self.__vocab))
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

    def probable_city(self, text):
        ngrams_list = ngrams(self.__c, text)
        most_probable = 0
        for ngram in ngrams_list:
            prob = self.prob(ngram[0], ngram[1])
            if prob > 0:
                most_probable += math.log(prob)
            else:
                return float('inf')
        return most_probable


################################################################################
# N-Gram Model with Interpolation
################################################################################

class NgramModelWithInterpolation(NgramModel):
    ''' An n-gram model with interpolation '''

    def __init__(self, c, k):
        super().__init__(c, k)
        self.__lambdas = []
        for i in range(c + 1):
            self.__lambdas.append(1 / (c + 1))

    def set_lambda(self, lam_vals):
        ''' sets the lambda values '''
        self.__lambda = lam_vals

    def update(self, text):
        ''' Updates the model n-grams based on text '''
        all_order_ngrams = []

        for char in text:
            if char not in super().get_vocab():
                self._NgramModel__vocab.update(char)

        for i in range(self._NgramModel__c + 1):
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
        for i in range(self._NgramModel__c + 1):
            prob1 = super().prob(context[len(context) - i:], char)
            prob = prob1 * self.__lambdas[i]
            probability += prob
        return probability


################################################################################
# Your N-Gram Model Experimentations
################################################################################
def NgramModel_tests():
    ''' NgramModel class tests '''
    print("----Running NgramModel tests----")
    m = NgramModel(1, 0)
    m.update('abab')
    print(m.get_vocab(), '(expected: {’b’, ’a’})')
    m.update('abcd')
    print(m.get_vocab(), '(expected: {’b’, ’a’, ’c’, ’d’}')
    print('WITHOUT INTERP')
    print(m.prob('a', 'a'), '(expected: 1.0)')
    print(m.prob('a', 'b'), '(expected: 1.0)')
    print('-----before k -----')
    print(m.prob('a', 'b'))
    print(m.prob('a', 'a'))
    print('-----after k -----')
    print(m.prob('a', 'b'))
    print(m.prob('a', 'a'))
    random.seed(1)
    print([m.random_char('') for i in range(25)])
    print(m.random_text(25))

    print('---Perplexity---')
    shakespeare = create_ngram_model(NgramModel, 'shakespeare_input.txt', 4)
    shakespeare.set_k(0)
    print("shakes perp to shakes: ", shakespeare.perplexity('shakespeare_input.txt'))
    print("shakes perp to sonnets: ", shakespeare.perplexity('shakespeare_sonnets.txt'))
    print("shakes perp to nyt: ", shakespeare.perplexity('nytimes_article.txt'))
    shakespeare.set_k(5)
    print("shakes perp to shakes: ", shakespeare.perplexity('shakespeare_input.txt'))
    print("shakes perp to sonnets: ", shakespeare.perplexity('shakespeare_sonnets.txt'))
    print("shakes perp to nyt: ", shakespeare.perplexity('nytimes_article.txt'))
    print('---Text Prediction---')
    print(shakespeare.prob("First", "Character"))
    print(shakespeare.random_text(300))
    print(m.perplexity('shakespeare_input.txt'), '(expected: 1.515716566510398)')
    print(m.perplexity('the quick brown fox jumped over the lazy sleeping dog'))


def NgramModelWithInterpolation_tests():
    ''' NgramModelWithInterpolation class tests '''
    print("----Running NgramModelWithInterpolation tests----")
    m = NgramModelWithInterpolation(1, 0)
    m.update('abab')
    print('WITH INTERP')
    print(m.prob('a', 'a'), '(expected: 0.25)')
    print(m.prob('a', 'b'), '(expected: 0.75)')
    a = NgramModelWithInterpolation(2, 1)
    a.update('abab')
    a.update('abcd')
    print(a.prob('~a', 'b'), '(expected: 0.4682539682539682)')
    print(a.prob('~c', 'd'), '(expected: 0.27222222222222225)')


def CitiesNgrams_tests():
    ''' gets the accuracy of the language model on predicting what country a city is in'''
    af = create_ngram_model(NgramModel, 'train/af.txt')
    cn = create_ngram_model(NgramModel, 'train/cn.txt')
    de = create_ngram_model(NgramModel, 'train/de.txt')
    fi = create_ngram_model(NgramModel, 'train/fi.txt')
    fr = create_ngram_model(NgramModel, 'train/fr.txt')
    ind = create_ngram_model(NgramModel, 'train/in.txt')
    ir = create_ngram_model(NgramModel, 'train/ir.txt')
    pk = create_ngram_model(NgramModel, 'train/pk.txt')
    za = create_ngram_model(NgramModel, 'train/za.txt')

    model_and_country = {af: 'af', cn: 'cn', de: 'de', fi: 'fi', fr: 'fr', ind: 'in',
                         ir: 'ir', pk: 'pk', za: 'za'}

    validation_files = {'val/af.txt': 'af', 'val/cn.txt': 'cn', 'val/de.txt': 'de', 'val/fi.txt': 'fi',
                        'val/fr.txt': 'fr', 'val/in.txt': 'in', 'val/ir.txt': 'ir', 'val/pk.txt': 'pk',
                        'val/za.txt': 'za'}

    validation_accuracy_counts = {'val/af.txt': 0, 'val/cn.txt': 0, 'val/de.txt': 0, 'val/fi.txt': 0,
                                  'val/fr.txt': 0, 'val/in.txt': 0, 'val/ir.txt': 0, 'val/pk.txt': 0,
                                  'val/za.txt': 0}

    for text in validation_files:
        file = open(text, 'r')
        validation_cities = file.read().splitlines()
        correct_country = validation_files[text]
        total_predictions = 0
        correct_predictions = 0
        for city in validation_cities:
            highest_prob = 0
            total_predictions += 1
            highest_prob_country = ''
            for model in model_and_country:
                prob = model.probable_city(city)
                if prob != float('inf'):
                    if highest_prob == 0 or (abs(prob) < abs(highest_prob)):
                        highest_prob = prob
                        highest_prob_country = model_and_country[model]
                elif highest_prob == 0:
                    highest_prob_country = model_and_country[model]
            if highest_prob_country == correct_country:
                correct_predictions += 1
        validation_accuracy_counts[text] = correct_predictions / total_predictions

    return validation_accuracy_counts


if __name__ == "__main__":
    print('-----------------')
    print('--RUNNING TESTS--')
    print('-----------------')
    #NgramModel_tests()
    #NgramModelWithInterpolation_tests()
    #print(CitiesNgrams_tests())
