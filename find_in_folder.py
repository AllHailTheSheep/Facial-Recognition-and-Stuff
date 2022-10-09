import face_recognition as fr
from shutil import copyfile as c

# Create an encoding of facial features that can be compared to other faces
picture_of_me = fr.load_image_file("path/to/folder")
my_face_encoding = fr.face_encodings(picture_of_me)[0]

# Iterate through all the pictures
for x in range(1, 304):
    # Construct the picture name and print it
    file_name = "folder/path/ (" + str(x) + ")" + ".jpg"
    print(file_name)

    # Load this picture
    new_picture = fr.load_image_file(file_name)

    end_name = "1 (" + str(x) + ")" + ".jpg"

    # Iterate through every face detected in the new picture
    for face_encoding in fr.face_encodings(new_picture):

        # Run the algorithm of face comaprison for the detected face, with 0.5 tolerance
        results = fr.compare_faces([my_face_encoding], face_encoding, 0.5)

        # Save the image to a seperate folder if there is a match
        if results[0] == True:
            c(file_name, "path/to/folder" + end_name)
