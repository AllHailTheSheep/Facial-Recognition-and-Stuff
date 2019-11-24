import face_recognition as fr
import PIL.ImageDraw as ImageDraw
import PIL.Image as Image

# Load Image
christi_image = fr.load_image_file("/home/deeplearning/Downloads/Known/Christi Fertich")

# Learn to recognize face
christi_encoding = fr.face_encodings(christi_image)[0]

# Open another image
joe_image = fr.load_image_file("/home/deeplearning/Downloads/Known/Joe Lynch")

# Learn Joe's face
joe_encoding = fr.face_encodings(joe_image)[0]

# Create array with known faces
known_encodings = [
    christi_encoding,
    joe_encoding
]

# Array to match encodings with names
known_names = [
    "Christi Fertich",
    "Joe Lynch"
]

# Load an image with an unknown face
unknown_image = fr.load_image_file("/home/deeplearning/Downloads/Unknown/29637181487_4cda12b951_o.jpg")

# Find all the faces and face encodings in the unknown image
face_locations = fr.face_locations(unknown_image)
face_encodings = fr.face_encodings(unknown_image, face_locations)

# Convert the image to a PIL-format image so that we can draw on top of it with the Pillow library
# See http://pillow.readthedocs.io/ for more about PIL/Pillow
pil_image = Image.fromarray(unknown_image)

# Create a Pillow ImageDraw Draw instance to draw with
draw = ImageDraw.Draw(pil_image)

# Loop through each face found in the unknown image
for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
    # See if the face is a match for the known face(s)
    matches = fr.compare_faces(known_encodings, face_encoding)

    name = "Unknown"

    # If a match was found in known_face_encodings, just use the first one.
    if True in matches:
        first_match_index = matches.index(True)
        name = known_names[first_match_index]

    # Draw a box around the face using the Pillow module
    draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

    # Draw a label with a name below the face
    text_width, text_height = draw.textsize(name)
    draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
    draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))


# Remove the drawing library from memory as per the Pillow docs
del draw

# Display the resulting image
pil_image.show()
