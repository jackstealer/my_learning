from datetime import datetime
import webbrowser
import requests
import speech_recognition as sr
import pyttsx3


greet_messages = ["hi", "hello", "hey", "hi there", "hey there"]
date_msgs = ["what's the date", "date", "tell me date", "today's date"]
time_msgs = ["what's the time", "time", "tell me time", "current time"]
weather_msgs = ["weather", "what's the weather", "tell me weather","temprature"]

engine = pyttsx3.init()
engine.setProperty("rate", 160)

def speak(text):
    engine.say(text)
    engine.runAndWait()

def listen():
    rec = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening....")
        audio = rec.listen(source)

    try:
        query = rec.recognize_google(audio)
        print("your query :", query)
        return query.lower()
    except Exception as ex:
        print("Can't catch that...")
        print("exception:", ex)
        return ""


WEATHER_API_KEY = "7d6397239df64b0b8c70351194079b5f"


def get_location():
    try:
        response = requests.get("http://ip-api.com/json/", timeout=5)
        data = response.json()
        city = data.get("city")
        country = data.get("country")

        if not city:
            return None, None

        return country, city
    except:
        return None, None


def get_weather(city):
    if not city:
        return "Could not detect your city."

    try:
        url = (
            "https://api.openweathermap.org/data/2.5/weather"
            f"?q={city}&appid={WEATHER_API_KEY}&units=metric"
        )

        response = requests.get(url, timeout=5)
        data = response.json()

        if response.status_code == 401:
            return "Invalid or inactive API key."

        if response.status_code == 404:
            return f"Weather not found for city: {city}"

        if response.status_code != 200:
            return "Weather service unavailable right now."

        weather = data["weather"][0]["description"]
        temp = data["main"]["temp"]
        feels_like = data["main"]["feels_like"]

        return f"Weather in {city}: {weather}, {temp}°C, feels like {feels_like}°C"

    except requests.exceptions.RequestException:
        return "Network error while fetching weather."


chat = True

while chat:
    msg = listen()

    if msg in greet_messages:
        response = "Hello, how are you?"
        print(response)
        speak(response)

    elif msg in date_msgs:
        response = str(datetime.now().date())
        print(response)
        speak(response)

    elif msg in time_msgs:
        response = datetime.now().strftime("%I:%M:%S")
        print(response)
        speak(response)

    elif msg in weather_msgs or "weather" in msg:
        country, city = get_location()
        response = get_weather(city)
        print(response)
        speak(response)

    elif "open" in msg:
        site = msg.split("open ")[-1]
        url = f"https://www.{site}.com"
        webbrowser.open(url)
        response = f"Opening {site}"
        print(response)
        speak(response)

    elif "location" in msg:
        country, city = get_location()
        if city:
            response = f"Your location is {city}, {country}"
        else:
            response = "Could not determine your location."
        print(response)
        speak(response)

    elif msg == "bye":
        response = "Goodbye"
        print(response)
        speak(response)
        chat = False

    else:
        response = "I can't understand"
        print(response)
        speak(response)
