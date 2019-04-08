import sys
import os
import scipy.stats
from nltk.corpus import wordnet_ic
from nltk.corpus import wordnet as wn
from utility import Utility


def maxScore(obj1, obj2, method):
    bestScore = 0
    bnc_ic = wordnet_ic.ic('ic-bnc.dat')
    for i in range(len(obj1)):
        for j in range(len(obj2)):
            pos1 = obj1[i].pos()
            pos2 = obj2[j].pos()
            if method == "path":
                if (pos1 != pos2) or pos1 == "s" or pos1 == "a":
                    continue
                score = obj1[i].path_similarity(obj2[j]) * 5
            elif method == "res":
                if (pos1 != pos2) or pos1 == "s" or pos1 == "a":
                    continue
                score = obj1[i].res_similarity(obj2[j], bnc_ic)
            elif method == "jcn":
                if (pos1 != pos2) or pos1 == "s" or pos1 == "a":
                    continue
                score = obj1[i].jcn_similarity(obj2[j], bnc_ic)
                if score == 1e300:
                    score = 1
                score = score * 5
            else:
                if (pos1 != pos2) or pos1 == "s" or pos1 == "a":
                    continue
                score = obj1[i].lin_similarity(obj2[j], bnc_ic) * 5
            if score != "None" and score > bestScore:
                bestScore = score
    return bestScore

def wordNet(data, method):
    newData = {}
    i = 1
    for key in data:
        w1, w2 = key.strip().split(":")
        obj1 = wn.synsets(w1)
        obj2 = wn.synsets(w2)
        score = maxScore(obj1, obj2, method)
        newData[key] = float(score)
        print(i)
        i += 1
    return newData
    

if __name__ == "__main__":
    read_file = "MTURK-771.csv"
    wordnet_method = "jcn"
    save_file = "out_wordnet_"+wordnet_method+".csv"

    utilityInstance = Utility()
    data = utilityInstance.readData(read_file)
    newData = wordNet(data, wordnet_method)
    utilityInstance.generateFile(save_file, newData)




