import matplotlib.pyplot as plt
import matplotlib.image as mpimg

file_paths = [
    ".//sir_labeling//plot_for_B=0.042.png",
    ".//sir_labeling//plot_for_B=0.062.png",
    ".//sir_labeling//plot_for_B=0.083.png",
]

# Create a figure and axes for the 1x3 grid
fig, axes = plt.subplots(1, 3, figsize=(15, 5))  # Adjust figsize as needed

# Iterate over the file paths and axes to display images
for i, ax in enumerate(axes.flat):
    if i < len(file_paths):  # Ensure we don't exceed the file list
        img = mpimg.imread(file_paths[i])  # Read the image
        ax.imshow(img)  # Display the image
        ax.axis('off')  # Turn off the axes
        # Add a title for each subplot with larger font size
        ax.set_title(f"SIR Dynamics for β={['0.042', '0.062', '0.083'][i]}", fontsize=14)

    else:
        ax.axis('off')  # Hide any extra axes

# Add a global title or labels if needed
# fig.suptitle("Combined Plots for Different B Values", fontsize=16, y=1.02)

# Adjust layout
plt.tight_layout()

# Save the figure with high quality
output_path = "C:\\Users\\venus\\Desktop\\spreading influence identification\\sir_labeling\\images\\plot_for_B_combined_plot.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')

# Display the plot
plt.show()
