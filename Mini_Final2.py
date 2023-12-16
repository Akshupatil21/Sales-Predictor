##########################################
#Required Packages
##########################################
import math
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

##########################################
#Function name: SalesPrediction
#Description: Predict the sales price based on advertisement
#Input:path of CSV file
#Output: Gives the sales price
#Date:06-06-2023
##########################################
def SalesPrediction(TV, Rd, Np):
    design = 50 * "-"

    df = pd.read_csv("advertising.csv", encoding="unicode_escape")
    print(design)
    print(df.head())

    df = df.rename(columns={"TV": "TV", "Radio": "Rd", "Newspaper": "Np"},inplace=False)

    print(design)
    print(df.head())

    Investment = df.loc[:, ["TV", "Rd", "Np"]]
    Investment.head()

    Investment["Total_Investment"] = Investment.sum(axis=1)

    Investment["Total_Investment"] = Investment.mean()             

    df["Sales"].mean()

    print(design, "\n")
    print("Information of dataset is given below:")
    print(design)
    df.info()
    print("The shape of dataset is:", df.shape)
    print(design)

    df.isna().sum()
    df.corr()

    # Visualize the relationship using scatterplot
    # TV vs sales
    f, ax = plt.subplots(figsize=(11, 9))
    plt.scatter(df["TV"], df["Sales"])
    plt.xlabel("TV")
    plt.ylabel("Sales")
    plt.title("Sales vs TV")
    plt.show()

    # Radio vs Sales
    f, ax = plt.subplots(figsize=(11, 9))
    plt.scatter(df["Rd"], df["Sales"])
    plt.xlabel("Radio")
    plt.ylabel("Sales")
    plt.title("Sales vs Radio")
    plt.show()

    # Newspaper vs Sales
    f, ax = plt.subplots(figsize=(11, 9))
    plt.scatter(df["Np"], df["Sales"])
    plt.xlabel("Newspaper")
    plt.ylabel("Sales")
    plt.title("Sales vs Newspaper")
    plt.show()

    df.describe().T


    X = df.drop("Sales",axis=1)
    Y = df["Sales"]

    X.head()
    Y.head()

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20)


    lm = LinearRegression()
    model = lm.fit(X_train, Y_train)

    Y_pred = lm.predict(X_test)

   
    f, ax = plt.subplots(figsize=(11, 9))
    plt.scatter(Y_pred, Y_test)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Actual vs Predicted")
    plt.show()

    df_comp = pd.DataFrame({'Actual Values': Y_test, 'Estimate': Y_pred})
    print("")

    MSE = mean_squared_error(Y_test, Y_pred)
    print("The mean squared difference between the estimated values and the actual value:", MSE)
    print("")

    
    print("R squared value of the model over training data:", model.score(X, Y))
    print("")

    new_data = pd.DataFrame({'TV': TV, 'Rd': Rd, 'Np': Np}, index=[1])

    Profit = model.predict(new_data)
    return Profit

##########################################
# function name:main
# Description: main function from where execution starts
# Date: 06-06-2023
##########################################
def main():
    print("---------------------------------Sales Predictor-------------------------------------------")

    print("Supervised Machine Learning")

    print("Linear Regression on advertising dataset")
    print("")

    print("How much money spend on advertisement of TV:")
    TV = float(input())

    print("How much money spend on advertisement of Radio:")
    Rd = float(input())

    print("How much money spend on advertisement of Newspaper:")
    Np = float(input())

    Profit = SalesPrediction(TV, Rd, Np)

    print("**********************************************************************")
    print("Prediction of Profit value in advertisement:", Profit)
    print("**********************************************************************")

##########################################
#Application Starter
##########################################
if __name__ == "__main__":
    main()
