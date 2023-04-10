import pyttsx3

engine = pyttsx3.init()
newVoiceRate = 125
engine.setProperty('rate',newVoiceRate)
while True:
    answer = input("Enter what you want the robot to say: \n")
    engine.say(answer)
    engine.runAndWait()