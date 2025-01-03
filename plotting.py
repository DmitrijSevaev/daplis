import matplotlib.pyplot as plt


def plot_per_file_times():
    number_of_nodes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
    time_per_file = [1.141, 3.166, 3.47, 4.086, 4.742, 5.018, 5.792, 5.954, 7.994, 6.873,
                     7.814, 8.131, 7.149, 6.564, 8.816, 7.865, 8.579, 8.184, 7.221]
    plt.figure(figsize=(8, 6))  # Width=12, Height=6
    # Create a bar plot with bars centered over x_values
    plt.bar(number_of_nodes, time_per_file, width=0.8, align='center')  # Align bars to the center
    # Set x-axis ticks to match x_values
    plt.xticks(number_of_nodes)
    # Add a comment to the first bar
    plt.text(number_of_nodes[0], time_per_file[0] + 0.5, 'Sequentially on PC', rotation=90, ha='center', va='bottom',
             color='red')
    # Add text labels to each bar
    for x, y in zip(number_of_nodes, time_per_file):
        plt.text(x, y + 0.1, f'{y:.2f}', ha='center', va='bottom')  # Adjust position slightly above the bar
    # Label the axes
    plt.xlabel('Number of Nodes')
    plt.ylabel('Seconds')
    # Add a title
    plt.title('Average time spent per file on Sunrise nodes')
    # Show the plot
    plt.show()


def plot_total_times():
    number_of_nodes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
    total_time = [41.2, 126.7, 69.4, 57.2, 47.5, 40.2, 40.6, 35.8, 40, 34.5,
                  31.3, 32.6, 28.6, 28.3, 28.5, 23.7, 25.8, 24.6, 21.7]

    plt.figure(figsize=(8, 6))  # Width=12, Height=6

    # Create a bar plot with bars centered over x_values
    plt.bar(number_of_nodes, total_time, width=0.8, align='center')  # Align bars to the center
    # Set x-axis ticks to match x_values
    plt.xticks(number_of_nodes)

    # Set y-axis limits to add space above the bars
    plt.ylim(0, max(total_time) + 20)  # Add some padding above the tallest bar

    # Add a comment to the first bar
    plt.text(number_of_nodes[0], total_time[0] + 15, 'Sequentially on PC', rotation=90, ha='center', va='bottom',
             color='red')
    # Add text labels to each bar
    for x, y in zip(number_of_nodes, total_time):
        plt.text(x, y + 1, f'{y:.1f}', rotation=90, ha='center', va='bottom')  # Adjust position slightly above the bar
    # Label the axes
    plt.xlabel('Number of Nodes')
    plt.ylabel('Seconds')
    # Add a title
    plt.title('Total time spent processing 40 files (3,1 GB) on Sunrise nodes')
    # Show the plot
    plt.show()


plot_per_file_times()
plot_total_times()
