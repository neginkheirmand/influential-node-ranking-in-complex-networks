import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file into a DataFrame
# from commit (8ad34a241ad8d7ca5a96e77f3d23129b3590928c) and commit message: (last part of cleaning the 3 precision data), the name of the file is NS/0.csv 

# df = pd.read_csv('./sir_labeling/prec_importance.csv')
df = pd.read_csv('./datasets/SIR_Results/NS/0.csv')

unique_values_count = df['SIR'].nunique()
print(f"There are {unique_values_count} different values in the SIR column.")
# There are 15 different values in the SIR column.


# Set the y-axis limit to a maximum of 1000
# plt.ylim(0, 1000)

# Plot the histogram of the SIR column
plt.hist(df['SIR'], bins=100, edgecolor='black')
# plt.title(f'Histogram IS - prec 3 (G: NS) - {unique_values_count} unique values ')
plt.title(f'Histogram IS - prec 6 (G: NS) - {unique_values_count} unique values ')
plt.xlabel('IS')
plt.ylabel('Frequency')
plt.show()