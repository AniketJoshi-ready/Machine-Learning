import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
def main():
    dataset=pd.read_csv("iris.csv")

    X= dataset.iloc[:,[0,1,2,3]].values

    model= KMeans(n_clusters=3,init="k-means++",n_init=10,random_state=42)
    model.fit(X)         
    print(X.shape)
    y_kmeans = model.fit_predict(X)

    print("values of y_kmeans")
    print(y_kmeans.shape)
    
    print("cluster of setosa :0")
    for i in range(10):
        print(X[i],y_kmeans[i])


    print("cluster of versicolor :1")
    for i in range(51,61):
        print(X[i],y_kmeans[i])    


    print("cluster of virginica :2")
    for i in range(101,111):
        print(X[i],y_kmeans[i])      

        


if __name__=="__main__":
    main()
  