# import the necessary packages
import imutils

def pyramid(image, scale=1.5, minSize=(30, 30)):
    # yield the original image
    yield image

    # keep looping over the pyramid
    while True:
        # compute the new dimensions of the image and resize it
        w = int(image.shape[1] / scale)
        image = imutils.resize(image, width=w)

        # if the resized image does not meet the supplied minimum
        # size, then stop constructing the pyramid
        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
            break

        # yield the next image in the pyramid
        yield image

def sliding_window(image, stepSize, windowSize):
    print("Stepsize: " + str(stepSize))
    print("window width: " + str(windowSize[0]))
    print("window height: " + str(windowSize[1]))
    for y in range(0, image.shape[0], windowSize[1] - stepSize):
        if image.shape[0] - y < windowSize[1]:
            windowSize[1] = image.shape[0] - y
            print("New windowsize: " + str(windowSize[1]))
        for x in range(0, image.shape[1], stepSize):
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])
