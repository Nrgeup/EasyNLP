import nltk
import enchant


if __name__ == '__main__':

    # split the sentence
    text = "Simple and  difficult at the same time - win matches at major " \
           "tournaments. England have no trouble racking up victories in " \
           "qualifiers. They have won 24 and drawn eight of their past 32 " \
           "but in world terms they have gone backwards because they do " \
           "not win when it matters."
    sentence = nltk.sent_tokenize(text)
    print(sentence)
    # >>> ['Simple and  difficult at the same time - win matches at major tournaments.', 'England have no trouble racking up victories in qualifiers.', 'They have won 24 and drawn eight of their past 32 but in world terms they have gone backwards because they do not win when it matters.']

    # split the word
    sentence = " we don't used Systran as provided by AltaVista (Babelfish)."
    print(nltk.word_tokenize(sentence))
    # >>> ['we', 'do', "n't", 'used', 'Systran', 'as', 'provided', 'by', 'AltaVista', '(', 'Babelfish', ')', '.']

    # check the spelling
    d = enchant.Dict("en_US")
    print(d.check("Hello"))
    # >>> True
    print(d.check("Helo"))
    # >>> False
    print(d.suggest("Helo"))
    # >>> ['we', 'do', "n't", 'used', 'Systran', 'as', 'provided', 'by', 'AltaVista', '(', 'Babelfish', ')', '.']

    # delete the stopwords
    string_tokenize = ["it", "is", "a", "beautiful", "girl"]
    english_stopwords = nltk.corpus.stopwords.words("english")
    string_tokenize = [word for word in string_tokenize if word not in english_stopwords]
    print(string_tokenize)
    # >>> ['beautiful', 'girl']

    # Stem
    st = nltk.stem.lancaster.LancasterStemmer()
    print(st.stem('stemmed'))
    # stem
    print(st.stem('stemming'))
    # stem

