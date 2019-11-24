import face_recognition as fr
import os
import shutil as s

known_faces = []

known_names = []

directory = input("What is the directory (to the folder) of known faces?")

final_dir = str(directory + " ")

for file in os.listdir(directory):
    known_faces.append(fr.face_encodings(fr.load_image_file(directory + file)))
    known_names.append(file)

# Iterate through all the pictures
for x in range(1, 304):
    # Construct the picture name and print it
    file_name = input("What the directory of the folder to be sorted?")+"/1 (" + str(x) + ")" + ".jpg"
    print(file_name)

    # Load this picture
    new_picture = fr.load_image_file(file_name)

    end_file_name = "1 (" + str(x) + ")" + ".jpg"

    # Iterate through every face detected in the new picture
    for face_encoding in fr.face_encodings(new_picture):

        # Run the algorithm of face comaprison for the detected face, with 0.5 tolerance
        results = fr.compare_faces([known_faces], face_encoding, 0.5)

        # Save the image to a seperate folder if there is a match
        if results[0] is True:
            s.copyfile(file_name, "/home/deeplearning/Pictures" + end_file_name)
            print(file_name)
