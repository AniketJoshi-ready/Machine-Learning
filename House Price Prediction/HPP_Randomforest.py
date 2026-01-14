import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,root_mean_squared_error,mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder





def Data_Cleaning(dataset):
    null_values=dataset.isnull().sum().sort_values(ascending=False)
    print(null_values)

    # filling numerical missing value with the median
    num_col=dataset.select_dtypes(include=np.number).columns
    for col in num_col:
        dataset[col].fillna(dataset[col].median(),inplace=True)

    # dropping irrelevant column
    dataset.drop(["Id","Location","Condition","Garage"],axis=1,inplace=True)
    #dataset = dataset.drop(["Garage", "Location"], axis=1)

    print(dataset)
    return dataset

def Exploratory_Data_Analysis(dataset):
    #  correlation heatmap
    numeric_df=dataset.select_dtypes(include=np.number)
    df_corr=numeric_df.corr()[["Price"]].sort_values(by='Price',ascending=False)
    plt.figure(figsize=(10,8))
    sns.heatmap(df_corr,annot=True,cmap='coolwarm',fmt='.2f')
    plt.title("feature correlation with price")
    plt.show()

    # Histplot
    sns.histplot(dataset['Price'],bins=5,kde=True,color='skyblue')
    plt.title("histogram plot for Price column")
    plt.show()

    #boxplot
    sns.boxplot(x=dataset['Location'],y=dataset['Price'],color='skyblue',width=0.5)
    #sns.stripplot(x=dataset["Location"],y=dataset['Price'],color='red',size=5,jitter=True)
    plt.title("boxplot for price ranges")
    plt.ylabel("price")
    plt.xlabel("Location")
    plt.show()

    # scatterplot : two numerical variables
    sns.scatterplot(x=dataset['Area'],y=dataset['Price'],hue=dataset['Location'])
    plt.title("relation of area with price ")
    plt.xlabel("area of house")
    plt.ylabel("price of house")
    plt.show()

    #pairplot:  Relationships between multiple numeric features
    # Why: Pairplots show scatterplots and distributions for multiple features. Useful for spotting patterns.
    selected_features=['Price','Area','Bedrooms','Bathrooms']
    sns.pairplot(dataset[selected_features])
    plt.title("relation between the features  ")
    plt.suptitle("pairplot:Price,Area,Bedrooms,Bathrooms")
    plt.show()

    # barchart:  Average Price by Condition
    # Why: Bar charts compare aggregated values across categories. Here, we compare average price by house condition.
    average_price_by_condition=dataset.groupby('Condition')['Price'].mean().sort_values()
    sns.barplot(x=average_price_by_condition.index,y=average_price_by_condition.values,palette='viridis')
    plt.title("Bar chart: avg house price by condition")
    plt.xlabel("conditions")
    plt.ylabel("avg house price")
    plt.show()

    #piechart:   Distribution of Garage availability
    # Why: Pie charts show proportions. We visualize how many houses have garages vs those that don’t.
    garage_count=dataset['Garage'].value_counts()
    plt.pie(garage_count,labels=garage_count.index,autopct='%1.1f%%' ,startangle=90)
    plt.title("piechart: distribution of garage availablity")
    plt.show()

def Data_Modelling(dataset):
    #le=LabelEncoder()
    #for col in ["Location","Condition","Garage"]:
        #dataset[col]=le.fit_transform(dataset[col])

    #dataset=pd.get_dummies(dataset,drop_first=True)
    #print("see after dummies:\n",dataset)

    #features
    x=dataset.drop("Price",axis=1) 
    print("see what we have features:\n",x)
    # labels 
    y=dataset["Price"]
    #print("see what we have as label:\n",y)

    #scalar=StandardScaler()
    #scaled_x=scalar.fit_transform(x)
    #print("scaled features:\n",scaled_x)

    X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.2,random_state=42)   

    # creating model by using ensemble ml algorithm 
    model=RandomForestRegressor(n_estimators=100,random_state=42)
    model.fit(X_train,Y_train)

    y_pred=model.predict(X_test)

    return Y_test,y_pred

def Evaluation(Y_test,y_pred):
    R2_SCORE=r2_score(Y_test,y_pred)*100
    RMSE_SCORE=root_mean_squared_error(Y_test,y_pred)

    print("How much variation in Price is explained by model is :",R2_SCORE)
    print("Average error in rupees between predicted and actual price is  :",RMSE_SCORE)


def main():
    df=pd.read_csv("House Price Prediction Dataset.csv")
    print(df)
    # shape of datyaframe
    print(df.shape)
    # display 1st 5 data
    print(df.head())
    # basic information about columns
    print(df.info())
    # statestical information
    print(df.describe())

    cl_df=Data_Cleaning(df)
    Exploratory_Data_Analysis(df)
    y_pred,Y_test=Data_Modelling(cl_df)
    Evaluation(y_pred,Y_test)


if __name__=="__main__":
    main() 