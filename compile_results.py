import matplotlib.pyplot as plt
import pandas as pd

def visualize_data(file_path):
    # Read the tab-separated data from the file
    df = pd.read_csv(file_path, sep='\t')

    grouped_df = df.groupby('Epsilon')
    print(df.head())
    print(df.tail())
    print(df.describe())
    print(df.corr())

    print(grouped_df.head())
    print(grouped_df.tail())
    print(grouped_df.describe())
    print(grouped_df.corr())

    # Extract the x and y values from the data
    x_values = [float(row[0]) for row in df.values]
    y_values = [float(row[1]) for row in df.values]
    
    # Create a scatter plot of the data
    plt.scatter(x_values, y_values)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Data Visualization')

    # Show the plot
    plt.show()
    # Calculate summary statistics
    # Create a new figure for summary visualizations
    plt.figure()

    # Plot a histogram of x values
    plt.subplot(2, 2, 1)
    plt.hist(x_values, bins=10)
    plt.xlabel('X')
    plt.ylabel('Frequency')
    plt.title('Histogram of X Values')

    # Plot a histogram of y values
    plt.subplot(2, 2, 2)
    plt.hist(y_values, bins=10)
    plt.xlabel('Y')
    plt.ylabel('Frequency')
    plt.title('Histogram of Y Values')

    # Plot a boxplot of x values
    plt.subplot(2, 2, 3)
    plt.boxplot(x_values)
    plt.xlabel('X')
    plt.ylabel('Value')
    plt.title('Boxplot of X Values')

    # Plot a boxplot of y values
    plt.subplot(2, 2, 4)
    plt.boxplot(y_values)
    plt.xlabel('Y')
    plt.ylabel('Value')
    plt.title('Boxplot of Y Values')

    # Adjust the layout of subplots
    plt.tight_layout()

    # Show the plot
    plt.show()



read_file = pd.read_csv('final_folder/final_big_m_results/results_dorefa_2.txt', sep='\t')
read_file.to_excel('dorefa_2_big_m_results.xlsx', index=None, header=True)
# Provide the file path to your tab-separated data file
file_path = 'results_dorefa_2.txt'
visualize_data(file_path)
