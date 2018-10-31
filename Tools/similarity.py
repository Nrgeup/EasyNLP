# http://www.ituring.com.cn/article/211389

import sys
import re
import math
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


def get_takenize(string):
    string = string.lower()
    string_tokenize = [word for word in word_tokenize(string)]
    english_punctuations = [",", ".", ":", ";", "?", "(", ")", "[", "]", '\'', '"', '=', '|',
                            "&", "!", "*", "@", "#", "$", "%", "|", "\\", "/", '{', '}']

    string_tokenize = [word for word in string_tokenize if word not in english_punctuations]
    english_stopwords = stopwords.words("english")
    string_tokenize = [word for word in string_tokenize if word not in english_stopwords]
    for word in string_tokenize:
        if re.match(r"^[0-9|\.|\\]+", word):
            string_tokenize.remove(word)
    # st = LancasterStemmer()
    return string_tokenize


def cal_cosine_similarity(string_x, string_y):
    string_x_takenize = get_takenize(string_x)
    string_y_takenize = get_takenize(string_y)
    vector_word_set = set(string_x_takenize + string_y_takenize)
    first_word_vector = []
    second_word_vector = []
    molecular = 0
    first_sum = 0
    second_sum = 0
    for word in vector_word_set:
        x1 = string_x_takenize.count(word)
        x2 = string_y_takenize.count(word)
        first_word_vector.append(x1)
        second_word_vector.append(x2)
        molecular += (x1 * x2)
        first_sum += (x1 * x1)
        second_sum += (x2 * x2)
    if first_sum == 0 or second_sum == 0:
        return 0
    cosine_similarity = float(molecular) / math.sqrt(float(first_sum * second_sum))
    return cosine_similarity


def cal_simple_common_words_similarity(string_x, string_y):
    string_x_takenize = get_takenize(string_x)
    string_y_takenize = get_takenize(string_y)
    max_length = max(len(string_x_takenize), len(string_y_takenize))
    common = 0
    for word in string_x_takenize:
        if word in string_y_takenize:
            common += 1
    return float(common/max_length)


def cal_jaccard_similarity(string_x, string_y):
    string_x_takenize = get_takenize(string_x)
    string_y_takenize = get_takenize(string_y)
    s_x = set(string_x_takenize)
    s_y = set(string_y_takenize)
    return float(len(s_x.intersection(s_y))/len(s_x.union(s_y)))


def cal_euclidean_similarity(string_x, string_y):
    string_x_takenize = get_takenize(string_x)
    string_y_takenize = get_takenize(string_y)
    vector_word_set = set(string_x_takenize + string_y_takenize)
    sum = 0
    for word in vector_word_set:
        x1 = string_x_takenize.count(word)
        x2 = string_y_takenize.count(word)
        sum += (x1-x2)*(x1-x2)
    return float(1/(1 + math.sqrt(sum)))


def cal_manhattan_similarity(string_x, string_y):
    string_x_takenize = get_takenize(string_x)
    string_y_takenize = get_takenize(string_y)
    vector_word_set = set(string_x_takenize + string_y_takenize)
    sum = 0
    for word in vector_word_set:
        x1 = string_x_takenize.count(word)
        x2 = string_y_takenize.count(word)
        sum += abs(x1-x2)
    return float(1/(1 + math.sqrt(sum)))

if __name__ == '__main__':
    a = "BLEU: Automatic evaluation by BLEU score"
    b = "We are given a source  BLEU sentence f = fJ1 = f1, . . . , fj , . . . , fJ , which is to be "
    print(cal_cosine_similarity(a, b))
    print(cal_simple_common_words_similarity(a, b))
    print(cal_jaccard_similarity(a, b))
    print(cal_euclidean_similarity(a, b))
    print(cal_manhattan_similarity(a, b))



