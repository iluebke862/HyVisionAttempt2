import pyautogui as gui
import tensorflow as tf
import matplotlib.pyplot as plt

# Load image
# screenshot = gui.screenshot()
# screenshot.show()
img = tf.io.read_file("/Users/iluebke862/Documents/GitHub/HyVisionAttempt2/images (1).jpeg")
img = tf.image.decode_jpeg(img, channels=1)
img = tf.image.convert_image_dtype(img, tf.float32)

# Add batch dimension: [1, H, W, 1]
img = tf.expand_dims(img, axis=0)

# Compute Sobel gradients
edges = tf.image.sobel_edges(img)
# Shape: [1, H, W, 1, 2] -> dy, dx

dy = edges[..., 0]
dx = edges[..., 1]

# Gradient magnitude
magnitude = tf.sqrt(tf.square(dx) + tf.square(dy))

# Normalize to [0, 1]
magnitude = magnitude / (tf.reduce_max(magnitude) + 1e-8)

# Display
plt.imshow(tf.squeeze(magnitude).numpy(), cmap="gray")
plt.axis("off")
plt.show()