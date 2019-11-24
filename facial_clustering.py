import os
import cv2
import face_recognition
import numpy as np
from sklearn.cluster import DBSCAN
from PIL import Image, ImageDraw
import shutil


def read_img(path):
    # Return a cv2 array given an image
    if os.path.isfile(path):
        return cv2.imread(path)
    # Love some good error handling
    else:
        raise ValueError('Path given does not seem to be valid: {}'.format(path))


# define the necessary stuff
original_dataset_path = input("Path to input directory of the images.")
sorted_dir_path = input("Path to output directory of the sorted images.")
detection_model = input("CNN or HOG?")

# First things first, we don't want to touch the originals in case something gets screwy, so lets go ahead and create
# a copy of them
print("INFO: Copying files...")
dataset_path = str(original_dataset_path) + "copy/"
if os.path.isdir(dataset_path) is True:
    shutil.rmtree(dataset_path, ignore_errors=True)
shutil.copytree(original_dataset_path, dataset_path)

# Grab a list of the image paths
imagePaths = []
for dirpath, dirnames, files in os.walk(dataset_path):
    for name in files:
        if name.lower().endswith(".jpg"):
            imagePaths.append(os.path.join(dirpath, name))


print("INFO: quantifying faces...")

# This will help in speeding up the process soon
eightk = (7680, 4320)
fourk = (3840, 2160)
qhd = (2460, 1440)
fullhd = (1920, 1080)
hd = (1280, 720)

# Start an empty array, to be added to soon
data = []
# Now we get to start the fun stuff
# Iterate over the paths
i = 0
for imagePath in imagePaths:
    if imagePath is None:
        raise Exception("Failed to grab the image paths. Check to make sure you entered the path right, including the "
                        "'/' at the end")
    # Open the image to resize it
    im = Image.open(imagePath)
    # This isn't necessary but makes it a whole heck ton faster
    imsize = im.size
    if eightk <= imsize:
        scale = 1/32
    if fourk <= imsize < eightk:
        scale = 1/16
    if qhd <= imsize < fourk:
        scale = 1/8
    if fullhd <= imsize < qhd:
        scale = 1/4
    if hd <= imsize < fullhd:
        scale = 1/2
    if imsize < hd:
        scale = 1
    # Load image
    image = read_img(imagePath)
    # Cut the size down to like 480p so it's faster. We get a warning as we're defining the variable scale inside an if
    # statement, and Python doesn't realize there's no case it can't be defined.
    image = cv2.resize(image, (0, 0), fx=scale, fy=scale)
    # OK so basically OpenCV uses BGR, but dlib uses RGB. Let's do a good ol' switcheroo.
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # So, I was having some issues here earlier with large image files, so we're gonna add some error handling
    if rgb is None:
        raise Exception("Image failed to load. Make sure you're entering the full path name, including the '/' at the"
                        " end")
    print("INFO:  " + str(round((i/len(imagePaths))*100, 2)) + "% done; processing image {}/{}:".format(i + 1, len(imagePaths)) + imagePath)

    # Now we detect the bounding boxes of the face (x, y)
    boxes = face_recognition.face_locations(rgb, model=detection_model)

    # Turn those pixels to a 128-d encoding
    encodings = face_recognition.face_encodings(rgb, boxes)

    i += 1

    # This is where we add to the data array
    d = [{"imagePath": imagePath, "loc": box, "encoding": enc}
         for (box, enc) in zip(boxes, encodings)]
    data.extend(d)

# Here, we cluster
encodings = [d["encoding"] for d in data]
clt = DBSCAN(metric="euclidean", n_jobs=-1)
# Perform DBSCAN on encodings.
clt.fit(encodings)


# Find all the unique faces, but only the ones with multiples (so it doesnt count all the outliers in -1)
labelIDs = np.unique(clt.labels_)
numUniqueFaces = len(np.where(labelIDs > -1)[0])

print("INFO: {} unique faces found".format(numUniqueFaces))

# Now we can iterate over the unique faces
for labelID in labelIDs:
    # Find all the indexes matching in the clustered data
    indexes = np.where(clt.labels_ == labelID)[0]
    # This will choose them at random. The min(len(imagePaths, len(indexes)) bit is important here for defining cluster
    # size
    indexes = np.random.choice(indexes, size=min(len(imagePaths), len(indexes)), replace=False)

    print("INFO: collecting " + str(len(indexes)) + " faces for face ID: {}...".format(labelID))

    # Check if the directory exists
    if os.path.isdir(sorted_dir_path + str(labelID)) is True:
        # If it is, delete it (this would presumably only result from previous runs of this program to the same sorted
        # directory).
        shutil.rmtree(sorted_dir_path + str(labelID), ignore_errors=True)
    # If it doesn't or did and has now been deleted, make it
    os.mkdir(sorted_dir_path + str(labelID))

    # So we need to only have this process occur once (the first time) here, so I'm going to use an integer counter.
    # It's not the most elegant, but it works.
    m = 0
    # Loop over the individual items in indexes
    for i in indexes:
        # So, sometime it is a little difficult to tell what face the program is clustering. Because of that let's add
        # one of the pictures with a box around the face that we are clustering around. We could have it analyze the
        # pictures once they're already sorted, but that seems like it would be slower. We only want this process to
        # occur the first time, though, as we only need one of these.
        # First, load the image
        image_location = data[i]["imagePath"]
        if len(indexes) > 1 and m == 0:
            image = data[i]["imagePath"]
            # First of all, read the image
            image = np.array(cv2.imread(image))
            # Remember how we did that switch way back from OpenCV to dlib? If we use that image, everybody is blue
            # as the blues and reds are switched. Let's un-switch them.
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            # I love PIL. We're reading an image from an array here.
            image = Image.fromarray(image)
            # Read the bounding box from the data we used
            (top, right, bottom, left) = data[i]["loc"]
            # Define PIL's drawing tool
            draw = ImageDraw.Draw(image)
            # Draw a rectangle (more of a square but whatever) on the picture, around the face.
            draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))
            # Pillow documentation wants us to delete the drawing after it's added to the image. I think it's mostly to
            # free up memory and avoid issues if you were doing multiple drawings, but if the documentation says to do
            # it, I'm going to do it.
            del draw
            # Save that image as clustered_face in the target directory
            image.save(sorted_dir_path + str(labelID) + "/clustered_face", format="jpeg")
        # Copy that image to the directory with it's label ID
        shutil.copy2(image_location, sorted_dir_path + str(labelID))
        # Add to int m so the whole clustered_face thing only happens once
        m += 1

# Last but not least, get rid of that directory we created in the first step, as it's just messy.
shutil.rmtree(dataset_path, ignore_errors=True)
