import requests
import urllib.request
import time
from bs4 import BeautifulSoup

url = 'https://he.flightaware.com/resources/registration/N840UA'
response = requests.get(url)

soup = BeautifulSoup(response.content, 'html.parser')
