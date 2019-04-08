#-*- coding:utf-8 -*-
import sys
import os
import scipy.stats
import csv


class Utility:
    def __init__(self):
        self.data = {}
        self.key = []
        
    def readData(self, filePath):
        csv_file = csv.reader(open(filePath, 'r'))
        for item in csv_file:
            [word1, word2, score] = item
            score = float(score)
            key = word1 + ":" + word2
            self.key.append(key)
            self.data[key] = score
        print("readData done!")
        return self.data
        
    def generateFile(self, writeFile, newData):
        out = open(writeFile, 'w', newline='')
        result = csv.writer(out, dialect='excel')

        originScore = []
        newScore = []
        for i in range(len(self.key)):
            key = self.key[i]
            w1, w2 = key.strip().split(":")
            result.writerow([w1, w2, str(newData[key])])
            originScore.append(self.data[key])
            newScore.append(newData[key])
        spearmanr = scipy.stats.spearmanr(originScore, newScore)[0]
        # result.write(str(spearmanr))
        # result.close()

