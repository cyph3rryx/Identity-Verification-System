# Identity Verification System

This is a Python-based Identity Verification System that uses multi-factor authentication to verify the identity of a user. The system currently supports facial recognition and fingerprint recognition as authentication factors, with the option to enable or disable each as needed.
Requirements

This system requires the following packages to be installed:

    OpenCV
    NumPy
    dlib
    face_recognition
    imutils
    scipy

## Usage

To use this system, follow these steps:

    1. Place images of known faces in the "known_faces" directory.
    
    2. Run the "main.py" script.
    
    3. Follow the on-screen prompts to complete authentication.

## Customization

This system can be customized by modifying the following variables in the "main.py" script:

    1. facial_recognition_enabled: Set to True to enable facial recognition authentication, or False to disable.
    
    2. fingerprint_recognition_enabled: Set to True to enable fingerprint recognition authentication, or False to disable.
    
    3. voice_recognition_enabled: Set to True to enable voice recognition authentication, or False to disable.
    
    4. max_attempts: Set the maximum number of authentication attempts allowed before authentication fails.
    
    5. threshold: Set the threshold for facial recognition authentication. A lower threshold will result in a more strict authentication process.

## Acknowledgements

This system was developed using the following resources:

    OpenCV
    NumPy
    dlib
    face_recognition
    imutils
    scipy

## License

This system is released under the MIT License.

### AUTHOR: Cyph3rRyx
