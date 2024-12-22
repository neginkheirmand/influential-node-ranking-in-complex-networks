import matplotlib.pyplot as plt
import matplotlib.image as mpimg

file_paths = [
    "./sir_labeling/images/diff_prec_plot_2.png",
    "./sir_labeling/images/diff_prec_plot.png"
    ]

# Create a figure and axes for the 1x5 grid
fig, axes = plt.subplots(1, 2, figsize=(15, 5))  # Adjust figsize as needed

# Iterate over the file paths and axes to display images
for i, ax in enumerate(axes.flat):
    if i < len(file_paths):  # Ensure we don't exceed the file list
        img = mpimg.imread(file_paths[i])  # Read the image
        ax.imshow(img)  # Display the image
        ax.axis('off')  # Turn off the axes
    else:
        ax.axis('off')  # Hide any extra axes

# Adjust layout
plt.tight_layout()

# Save the figure with high quality
output_path = "C:\\Users\\venus\\Desktop\\spreading influence identification\\sir_labeling\\images\\combined_plot.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')

# Display the plot
plt.show()
