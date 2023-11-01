import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


def load_and_preprocess_data(filename):
    data = pd.read_csv(filename)
    data['dateRep'] = pd.to_datetime(data['dateRep'])
    return data

def select_features_and_target(data):
    X = data[['day', 'month', 'year']].values
    y = data['deaths'].values
    return X, y


def build_linear_regression_model(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model

def predict_max_death_date(data, model):
    predicted_deaths = model.predict(data[['day', 'month', 'year']].values)
    max_death_index = predicted_deaths.argmax()
    max_death_date = data.iloc[max_death_index]['dateRep']
    max_death_deaths = predicted_deaths[max_death_index]
    return max_death_date, max_death_deaths

def visualize_data(data, predicted_deaths):
    plt.figure(figsize=(10, 6))
    plt.scatter(data['dateRep'], data['deaths'], label='Actual Deaths')
    plt.plot(data['dateRep'], predicted_deaths, 'r', label='Predicted Deaths')
    plt.xlabel('Date')
    plt.ylabel('Deaths')
    plt.title('Actual vs. Predicted Deaths')
    plt.legend()
    plt.show()


def main():
    filename = 'Covid_19_cases4.csv'  
    data = load_and_preprocess_data(filename)
    X, y = select_features_and_target(data)
    model = build_linear_regression_model(X, y)
    max_death_date, max_death_deaths = predict_max_death_date(data, model)

    print(f"Date with the maximum predicted deaths: {max_death_date}")
    print(f"Predicted deaths on {new_date}: {predicted_deaths[0]}")

    visualize_data(data, model.predict(X))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ... (your previous code)

def exploratory_data_analysis(data):
    # Summary statistics
    summary = data.describe()
    print("Summary Statistics:")
    print(summary)

    # Data distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(data['deaths'], kde=True)
    plt.title('Distribution of Deaths')
    plt.xlabel('Deaths')
    plt.ylabel('Frequency')
    plt.show()

    # Correlation matrix
    correlation_matrix = data.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()

    # Time series analysis
    daily_deaths = data.groupby('dateRep')['deaths'].sum()
    plt.figure(figsize=(12, 6))
    daily_deaths.plot()
    plt.title('Daily Deaths Over Time')
    plt.xlabel('Date')
    plt.ylabel('Daily Deaths')
    plt.show()

    # Box plots for categorical variables
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='continentExp', y='deaths', data=data)
    plt.title('Deaths by Continent')
    plt.xlabel('Continent')
    plt.ylabel('Deaths')
    plt.xticks(rotation=45)
    plt.show()

def statistical_analysis(data):
    # Placeholder for statistical analysis
    # You can perform statistical tests, hypothesis testing, regression analysis, etc.
    
    # Example: Linear Regression
    X = data[['day', 'month', 'year']]
    y = data['deaths']
    X = sm.add_constant(X)  # Add a constant for the intercept
    model = sm.OLS(y, X).fit()
    
    # Print regression summary
    print("Linear Regression Summary:")
    print(model.summary())



def main():
    filename = 'Covid_19_cases4.csv'  
    data = load_and_preprocess_data(filename)
    X, y = select_features_and_target(data)
    model = build_linear_regression_model(X, y)
    max_death_date, max_death_deaths = predict_max_death_date(data, model)

    print(f"Date with the maximum predicted deaths: {max_death_date}")
    print(f"Predicted deaths on {max_death_date}: {max_death_deaths}")

    exploratory_data_analysis(data)
    statistical_analysis(data)

def exploratory_data_analysis(data):
    # ... (previous code)

    # Pairplot to explore relationships between variables
    sns.pairplot(data[['deaths', 'cases', 'popData2019']], kind='scatter')
    plt.suptitle('Pairplot of Deaths, Cases, and Population Data')
    plt.show()

    # Time series line plots for selected continents
    continents_to_plot = ['Europe', 'Asia', 'Africa', 'North America']
    plt.figure(figsize=(12, 6))
    for continent in continents_to_plot:
        subset = data[data['continentExp'] == continent]
        daily_deaths = subset.groupby('dateRep')['deaths'].sum()
        daily_deaths.plot(label=continent)

    plt.title('Daily Deaths Over Time for Selected Continents')
    plt.xlabel('Date')
    plt.ylabel('Daily Deaths')
    plt.legend()
    plt.show()

    # Box plots for categorical variables
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='continentExp', y='deaths', data=data)
    plt.title('Deaths by Continent')
    plt.xlabel('Continent')
    plt.ylabel('Deaths')
    plt.xticks(rotation=45)
    plt.show()

    # Pairwise correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
    plt.title('Pairwise Correlation Heatmap')
    plt.show()

    # Scatterplot of cases and deaths
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=data, x='cases', y='deaths', hue='continentExp', alpha=0.6)
    plt.title('Scatterplot of Cases vs. Deaths')
    plt.xlabel('Cases')
    plt.ylabel('Deaths')
    plt.legend(title='Continent')
    plt.show()

def main():
    # ... (previous code)

if _name_ == "_main_":
    main()


if _name_ == "_main_":
    main()
