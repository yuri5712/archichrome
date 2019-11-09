import numpy as np

clusterNum = 5
centroids = np.random.randint(0, 255, (clusterNum, 3))
print ("centroids:", centroids)

def printN():
    print ("aaa")

if __name__ == '__main__':
    aaa = printN()