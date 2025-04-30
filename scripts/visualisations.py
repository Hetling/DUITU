import matplotlib.pyplot as plt

def show_image_and_label(image, label, segment = 17): # segment is the id of the class to show, 17 is road
    print('Image shape:', image.shape)
    print('Label shape:', label.shape)
    plt.subplot(1, 2, 1)
    plt.imshow(image.permute(1, 2, 0))  # Convert from (C, H, W) to (H, W, C)
    plt.title('Image')
    plt.axis('off')

    #show the label
    plt.subplot(1, 2, 2)
    plt.imshow(label[segment], cmap='gray')  # Assuming label is a single channel
    plt.show()