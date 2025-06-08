import matplotlib.pyplot as plt
import cv2

def img_debug(title="Image Comparison", **images):
    """
    Displays any amount of images with their corresponding titles.

    Parameters:
        title (str): The main title for the plot.
        **images (dict): A dictionary where keys are image titles (str)
                         and values are the image data (np.array).
                         Example: img1=image_data1, img2=image_data2
    """
    if not images:
        print("No images provided to display.")
        return

    num_images = len(images)
    fig, axes = plt.subplots(1, num_images, figsize=(5 * num_images, 5))
    fig.suptitle(title, fontsize=16)

    # Ensure axes is an array even if there's only one image
    if num_images == 1:
        axes = [axes]

    for i, (img_title, img_data) in enumerate(images.items()):
        # Attempt to convert to RGB if the image has 3 channels,
        # otherwise display as is (e.g., grayscale for masks).
        if len(img_data.shape) == 3 and img_data.shape[2] == 3:
            try:
                axes[i].imshow(cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB))
            except Exception: # Fallback for non-BGR images that are 3 channel
                 axes[i].imshow(img_data)
        else:
            axes[i].imshow(img_data, cmap='gray') # Assume grayscale for 1-channel images

        axes[i].set_title(img_title)
        axes[i].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
