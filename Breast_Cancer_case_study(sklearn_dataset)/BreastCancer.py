import pandas as pd 
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def ExploratoryDataAnalysis(dataframe):
    print("statestical summery of data :\n")
    print(dataframe.describe())

    # data visualisation:
    print("Feature Correlation on heatmap:")
    plt.figure(figsize=(12,10))
    corr = dataframe.corr()   # correlation matrix
    sns.heatmap(corr, cmap="coolwarm", center=0, linewidths=0.5)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.show()

    print("now feature correlation with targets \n we will show with only top 10 feature :")
    target_corr = corr["target"].drop("target").sort_values(ascending=False)
    print("\nTop 10 features correlated with target:")
    print(target_corr.head(10))

    """Positive correlation → feature values are higher in benign cases.
       Negative correlation → feature values are higher in malignant cases.
       Closer to 0 → feature is not very useful for classification."""

def Data_preprocessing(feature,y):
    scalar=StandardScaler()
    x_scaled=scalar.fit_transform(feature)
    print("After scaling: X_scaled shape =", x_scaled.shape, " y shape =", y.shape)

    # splitting data into train and test :
    X_train,X_test,Y_train,Y_test=train_test_split(x_scaled,y, test_size=0.2,random_state=42,stratify=y)

    return X_train,X_test,Y_train,Y_test

def Model_Building(X_train,X_test,Y_train,Y_test):
    model = LogisticRegression()   # if need max_iter=5000, random_state=42 try karo ek baar 
    model.fit(X_train, Y_train)  # model build on training set
    
    model_predicted = model.predict(X_test) # model build on testing set

    print("model build on training set\nmodel build on testing set")

    return model_predicted

def Evaluation(target_test,model_predicted):
    print("Accuracy:", accuracy_score(target_test,model_predicted)*100)
    print("\nConfusion Matrix:\n", confusion_matrix(target_test,model_predicted))
    print("\nClassification Report:\n", classification_report(target_test,model_predicted))  




def main():
    data=load_breast_cancer()
    df=pd.DataFrame(data.data,columns=data.feature_names) # for me : data is variable container and other .data and .feature_name they

    print(df)
    df["target"]=data.target

    #df["label"]=df["target"].map({0: "malignant", 1: "benign"}) for just reference 
    #print(df.info())
    #print(df[67:108])
   
    # features:
    x=df.drop(columns=["target"])
    # label:
    y=df["target"]
    
    ExploratoryDataAnalysis(df)
    X_train,X_test,Y_train,Y_test=Data_preprocessing(x,y)
    y_pred=Model_Building(X_train,X_test,Y_train,Y_test)
    Evaluation(Y_test,y_pred)

    print("THE END OF PROGRAM ( BREAST CANCER CASE STUDY )")
if __name__=="__main__":
    main()
    