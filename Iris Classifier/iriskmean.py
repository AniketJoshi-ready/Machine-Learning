import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
def main():
    dataset=pd.read_csv("iris.csv")

    X= dataset.iloc[:,[0,1,2,3]].values

    print(X)

    wcss=[]

    for k in range(1,11):           
        model= KMeans(n_clusters=k,init="k-means++",n_init=10,random_state=42)
        model.fit(X)
        print(model.inertia_)  

        wcss.append(model.inertia_)



    plt.plot(range(1,11),wcss,marker="o")
    plt.title("elbo methods for kmeans ")
    plt.xlabel("no.of clusters K")
    plt.ylabel("wcss within cluster sum of square")
    plt.grid(True)
    plt.show()             


if __name__=="__main__":
    main()
  