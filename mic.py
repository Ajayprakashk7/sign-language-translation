import speech_recognition as sr
import cv2
import time
from string import ascii_lowercase




# Define the letters to detect
detect=['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']



# Define the file path for the images
IMG_PATH = "audio/{}.jpg"


# Define the mapping between letters and image file names
LETTERS = {letter: str(index) for index, letter in enumerate(ascii_lowercase, start=1)}
LETTERS.update({letter.upper(): str(index) for letter, index in LETTERS.items()})




# Define a function to get the position of a letter in the alphabet
def alphabet_position(text):
    text = text.lower()
    numbers = [LETTERS[character] for character in text if character in LETTERS]
    return ' '.join(numbers)




# Define a function to combine the images of each letter in a word into a single image for that word
def combine_images(word):
    images = []
    for letter in word:
        if letter != " ":
            # Get the image file name for the current letter
            img_path = IMG_PATH.format(LETTERS[letter])


            # Read in the image
            image = cv2.imread(img_path)
            images.append(image)


    # Combine the images horizontally
    combined_image = cv2.hconcat(images)
    return combined_image




# Initialize the speech recognizer
r = sr.Recognizer()


# Initialize the microphone
speech = sr.Microphone(device_index=1)


auto_close = True  # Flag to determine if the image should automatically close


while True:
    with speech as source:
        # Adjust the ambient noise and listen for audio input
        print("Say something...")
        r.adjust_for_ambient_noise(source)
        audio = r.listen(source)


    try:
        # Attempt to recognize the speech
        recog = r.recognize_google(audio, language='en-US')


        print("You said:", recog)


        # Split the recognized speech into words
        words = recog.split()


        # Loop over each word in the recognized speech
        for word in words:
            # Combine the images for each letter in the word
            combined_image = combine_images(word)


            # Display the image for the word
            cv2.imshow("Image", combined_image)


            # Wait for 3 seconds or until a key is pressed
            if auto_close or " " in word:
                key = cv2.waitKey(1500)
            else:
                key = cv2.waitKey(0)


            # If the s and 1 keys are pressed together, stop the program
            if key == ord('s') and cv2.waitKey(0) == ord('1'):
                cv2.destroyAllWindows()
                break


            # Toggle the auto_close flag when spacebar is pressed
            if key == ord(' '):
                auto_close = not auto_close


            # Destroy the window before proceeding to the next word
            cv2.destroyAllWindows()


            # Wait for a short time before proceeding to the next word
            time.sleep(0.1)


    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")


    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))


    # Wait for a short time before attempting to recognize speech again
    time.sleep(0.1)
