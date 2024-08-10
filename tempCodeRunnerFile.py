def prep_images(directory):
    """Read images into list and resize to match training images."""
    
    images = []

    for image in os.listdir(directory):
        img = cv2.imread(os.path.join(directory, image))
        # Match training image size of 30x30
        img = cv2.resize(img, (30, 30), interpolation=cv2.INTER_AREA)
        images.append(
            {'name': image, 'array': img}
        )

    return images