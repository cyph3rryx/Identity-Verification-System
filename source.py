import face_recognition

# Store the known faces and their names
known_faces = {
    "Ryx": face_recognition.load_image_file("ryx.jpg"),
    "Gojo": face_recognition.load_image_file("gojo.jpg"),
    "Kira": face_recognition.load_image_file("kira.jpg")
}

# Encode the known faces using face_recognition library
known_encodings = {}
for name, face in known_faces.items():
    known_encodings[name] = face_recognition.face_encodings(face)[0]

# Capture the image of the person who wants to enter
unknown_image = face_recognition.load_image_file("unknown.jpg")
unknown_encoding = face_recognition.face_encodings(unknown_image)[0]

# Compare the unknown face to the known faces
results = face_recognition.compare_faces(list(known_encodings.values()), unknown_encoding)

# Get the names of the known faces
names = list(known_encodings.keys())

# Display the results
if True in results:
    # Get the name of the matched face
    match_index = results.index(True)
    name = names[match_index]
    print(f"Welcome, {name}!")
else:
    # Display error message if the face doesn't match any of the known faces
    print("Sorry, your identity cannot be verified.")
