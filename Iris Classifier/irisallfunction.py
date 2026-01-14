import pandas as pd
from sklearn.model_selection import train_test_split


def loaddata(file_path):
    df=pd.read_csv(file_path)
    print("dataset successfully loaded in memory ")
    return df
def getinformation(df):
    print("shapeof dataset :",df.shape)
    print("column :",df.columns)
    print("missing values :",df.isnull().sum())

def encodedata(df):
    df["variety"]=df["variety"].mapmap({"Setosa":0,"Versicolor":1,"Virginica":2})
    return df

def split_feature_target(df):
    X=df.drop("variety",axis=1)
    Y=df["variety"]

    return X,Y


def split(X,Y,size=0.2):
    return train_test_split(X,Y,test_size=size)

def main():
    data=loaddata("iris.csv")
    
    print(data.head())
    getinformation(data)

    print("after encodeing data : ")
    data =encodedata(data)

    print(data.head())

    independent, dependent =split_feature_target(data)
    print(independent)
    print(dependent)


    X_train,X_test,Y_train,Y_test =split(independent,dependent,0.3)


    print(X_train.shape)
    print(X_test.shape)
    print(Y_train.shape)
    print(Y_test.shape)




  
if __name__=="__main__":
    main()    