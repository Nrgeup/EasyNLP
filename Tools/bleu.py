import nltk


def english_segmentation(sentence):
    """"""
    # Converted to lowercase letters
    sentence = sentence.lower()
    # Separate words and token
    sentence = nltk.tokenize.word_tokenize(sentence)
    # delete the token
    english_punctuations = [",", ".", ":", ";", "?", "(", ")", "[", "]", '\'', '"', '=', '|',
                            "&", "!", "*", "@", "#", "$", "%", "|", "\\", "/", '{', '}']
    sentence = [e for e in sentence if e not in english_punctuations]
    return sentence

def get_bleu_score(hypothesis, reference):
    hypothesis = english_segmentation(hypothesis)
    reference = english_segmentation(reference)
    weights = (0.25, 0.25, 0.25, 0.25)
    bleu_score = nltk.translate.bleu_score.sentence_bleu([reference], hypothesis, weights)
    return bleu_score

if __name__ == '__main__':
    Ref = "It is the guiding principle which guarantees" \
          " the military forces always being under the command of the Party."
    Candidate = "It is the practical guide for the army" \
                " always to heed the directions of the party."
    print(get_bleu_score(Candidate, Ref))
