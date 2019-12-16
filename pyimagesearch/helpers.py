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

def sliding_window(image, x_stepSize, y_stepSize, windowSize):
    print("x_Stepsize: " + str(x_stepSize))
    print("y_Stepsize: " + str(y_stepSize))
    print("window width: " + str(windowSize[0]))
    print("window height: " + str(windowSize[1]))
    original_windowSizeX = windowSize[0]
    for y in range(0, image.shape[0], windowSize[1] - y_stepSize):
        # windowSize[0] = original_windowSizeX
        # if y + windowSize[1] > image.shape[1]:
        #     windowSize[1] = image.shape[1] - y
        #     # print("New y_windowsize: " + str(windowSize[1]))
        for x in range(0, image.shape[1], x_stepSize):
            # if x + windowSize[0] > image.shape[0]:
            #     windowSize[0] = image.shape[0] - x
            # # print("New x_windowsize: " + str(windowSize[0]))
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])
