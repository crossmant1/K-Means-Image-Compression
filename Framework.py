from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider, Button


# ImageCompressor class definition
class ImageCompressor:
    def __init__(self, image_path):
        self.original_image = np.array(Image.open(image_path)) / 255.0  # Normalize to [0, 1]
        self.kmeans_images = {}  # Dictionary to store k-means results
        self.last_k = 0  # Track the last k-value used

    def kmeans(self, num):
        if num in self.kmeans_images:
            return  # if the result is already computed, it skips the calculation

        if num == 0:
            self.kmeans_images[0] = self.original_image
            return

        # flatten image for clustering height * width, channels
        height, width, channels = self.original_image.shape
        flat_image = self.original_image.reshape(-1, channels)

        # initialize the centers
        centroids = np.linspace(0, 1, num)[:, np.newaxis] * np.ones((1, channels))

        # K-means clustering
        max_iterations = 50
        threshold = 0.01
        for _ in range(max_iterations): # iterate through the number of iterations
            distances = np.linalg.norm(flat_image[:, np.newaxis] - centroids, axis=2) # computes the distance between each pixel and the center
            labels = np.argmin(distances, axis=1) # assigns each pixel to the closest center
            old_centroids = centroids.copy() # copies the centers array
            for i in range(num): # loops through each cluster
                cluster_pixels = flat_image[labels == i] # gets all the pixels assigned to i
                if len(cluster_pixels) > 0: # checks for pixels assigned to i
                    centroids[i] = np.mean(cluster_pixels, axis=0) # updates the center of i to the average of its assigned pixels
            if np.all(np.abs(old_centroids - centroids) < threshold): # stops the loops when the centers are not changing significantly
                break

        # Reconstruct compressed image
        compressed_image = centroids[labels].reshape(height, width, channels)
        self.kmeans_images[num] = compressed_image
        self.last_k = num  # Update last k-value

    def get_image(self):
        if self.last_k not in self.kmeans_images:
            self.kmeans(self.last_k)  # Compute if not already done
        return self.kmeans_images[self.last_k]


# List of image filenames
image_filenames = ["test1.jpg", "test2.jpg", "test3.jpg", "test4.jpg", "test5.jpg"]

# Create list of ImageCompressor objects
image_compressor_list = [ImageCompressor(filename) for filename in image_filenames]

# Initialize with the first image
current_image_index = 0
current_compressor = image_compressor_list[current_image_index]
current_compressor.kmeans(0)  # Initialize with original image

# Create the plot
fig, ax = plt.subplots(figsize=(8, 8))
plt.subplots_adjust(bottom=0.25)
ax.set_title("K-Means Image Compression")
ax.axis('off')
img_display = ax.imshow(current_compressor.get_image())

# Add slider for k-value control
ax_slider = plt.axes([0.25, 0.15, 0.5, 0.03])
slider = Slider(ax_slider, 'Clusters', 0, 64, valinit=0, valstep=1)


# Update function for slider
def update(val):
    k = int(slider.val)
    current_compressor.kmeans(k)  # Compute or retrieve k-means result
    img_display.set_data(current_compressor.get_image())
    fig.canvas.draw_idle()


# Function to change images
def change_image(direction):
    global current_image_index, current_compressor
    if direction == 'next':
        current_image_index = (current_image_index + 1) % len(image_compressor_list)
    elif direction == 'prev':
        current_image_index = (current_image_index - 1) % len(image_compressor_list)

    current_compressor = image_compressor_list[current_image_index]
    slider.set_val(current_compressor.last_k)  # Update slider to last k-value
    img_display.set_data(current_compressor.get_image())
    fig.canvas.draw_idle()


# Add navigation buttons
ax_prev = plt.axes([0.1, 0.05, 0.1, 0.075])
ax_next = plt.axes([0.8, 0.05, 0.1, 0.075])
button_prev = Button(ax_prev, 'Previous')
button_next = Button(ax_next, 'Next')

# Connect button and slider events
button_prev.on_clicked(lambda event: change_image('prev'))
button_next.on_clicked(lambda event: change_image('next'))
slider.on_changed(update)

# Show the plot
plt.show()