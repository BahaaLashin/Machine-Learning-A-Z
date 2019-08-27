import numpy as np
import matplotlib.pyplot as plt

dataset = np.array([[1,2,'r'],[2,2,'r'],[5,6,'b'],[2,2.5,'r'],[6,7,'b'],[5,4,'r'],[1,1.5,'r'],[4,4,'b'],[4,2.5,'b'],[.5,2,'r'],[1,1,'r']])


point = np.array([3,5])

def get_knn(dataset,point,k=3):
    distances = []
    categorices = dataset[:,-1]
    for i in dataset:
        data = float(i[0]) - float(point[0]) + float(i[1]) - float(point[1])

        distance = np.sqrt(np.power(data,2))
        distances.append([distance,i[2]])
    distances = sorted(distances)
    distances = np.array(distances)
    firstk = distances[:k]

    print(firstk)
    plt.scatter(point[0],point[1],color='black')
    for i in dataset:
        if i[2] == 'r':
            plt.scatter(i[0], i[1], color='red')
        else:
            plt.scatter(i[0], i[1], color='blue')

    plt.show()






get_knn(dataset,point)

