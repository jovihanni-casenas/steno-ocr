import speech_recognition as sr

# obtain audio from the microphone
r = sr.Recognizer()
mic = sr.Microphone()

print("Start Talking!")

while True:
    try:
        with mic as source:
            r.adjust_for_ambient_noise(source)
            audio = r.listen(source, timeout=3)
        words = r.recognize_google(audio)
        print(words)
    except:
        print("Error")