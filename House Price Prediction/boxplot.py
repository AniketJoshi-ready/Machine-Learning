
# Creating boxplot for exam scores with outliers highlighted
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Create the dataset
data = pd.DataFrame({
    "Student": ['A', 'B', 'C', 'D', 'E'],
    "Score": [45, 88, 95, 60, 100]
})

# Set plot style
sns.set(style="whitegrid")

# Create the boxplot
plt.figure(figsize=(6, 8))
sns.boxplot(y=data["Score"], color="skyblue", width=0.3)

# Add individual data points
sns.stripplot(y=data["Score"], color="black", size=8, jitter=False)

# Labeling
plt.title("Distribution of Exam Scores with Outliers", fontsize=14)
plt.ylabel("Score", fontsize=12)
plt.xticks([])  # Hide x-axis ticks since we only have one box
plt.show()
"""
# Save the plot
import os
if not os.path.exists("/mnt/data"):
    os.makedirs("/mnt/data")
plt.savefig("/mnt/data/exam_scores_boxplot.png")
plt.close()"""
