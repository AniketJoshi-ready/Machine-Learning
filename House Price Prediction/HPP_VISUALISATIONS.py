import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("House Price Prediction Dataset.csv")

# Set plot style
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

# 1️⃣ Boxplot: Price distribution across Locations
# Why: Boxplots are great for spotting outliers and comparing price spread across categories like Location
sns.boxplot(x='Location', y='Price', data=df)
plt.title("Boxplot: House Price Distribution by Location")
plt.xlabel("Location")
plt.ylabel("Price")
plt.xticks(rotation=45)
plt.show()

# 2️⃣ Scatterplot: Area vs Price
# Why: Scatterplots show relationships between two numeric variables. Here, we check if larger houses cost more.
sns.scatterplot(x='Area', y='Price', data=df, hue='Location')
plt.title("Scatterplot: Area vs Price")
plt.xlabel("Area (sq ft)")
plt.ylabel("Price")
plt.show()

# 3️⃣ Histplot: Distribution of House Prices
# Why: Histograms show how data is distributed. This helps us see if prices are skewed or normally distributed.
sns.histplot(df['Price'], bins=30, kde=True, color='skyblue')
plt.title("Histplot: Distribution of House Prices")
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.show()

# 4️⃣ Heatmap: Correlation between numeric features
# Why: Heatmaps visualize correlations. We use it to find which features are most related to Price.
numeric_df = df.select_dtypes(include='number')
corr_matrix = numeric_df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Heatmap: Feature Correlation Matrix")
plt.show()

# 5️⃣ Pairplot: Relationships between multiple numeric features
# Why: Pairplots show scatterplots and distributions for multiple features. Useful for spotting patterns.
selected_features = ['Price', 'Area', 'Bedrooms', 'Bathrooms']
sns.pairplot(df[selected_features])
plt.suptitle("Pairplot: Price, Area, Bedrooms, Bathrooms", y=1.02)
plt.show()

# 6️⃣ Bar Chart: Average Price by Condition
# Why: Bar charts compare aggregated values across categories. Here, we compare average price by house condition.
avg_price_by_condition = df.groupby('Condition')['Price'].mean().sort_values()
sns.barplot(x=avg_price_by_condition.index, y=avg_price_by_condition.values, palette='viridis')
plt.title("Bar Chart: Average House Price by Condition")
plt.xlabel("Condition")
plt.ylabel("Average Price")
plt.show()

# 7️⃣ Pie Chart: Distribution of Garage availability
# Why: Pie charts show proportions. We visualize how many houses have garages vs those that don’t.
garage_counts = df['Garage'].value_counts()
plt.pie(garage_counts, labels=garage_counts.index, autopct='%1.1f%%', startangle=90, colors=['lightgreen', 'lightcoral'])
plt.title("Pie Chart: Garage Availability")
plt.axis('equal')  # Ensures pie is a circle
plt.show()
