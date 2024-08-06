#!/usr/bin/python3
""" APIs """
import requests
import sys
from datetime import datetime


def get_user_location(url):
    """ get user location """
    response = requests.get(url)
    if response.status_code == 404:
        print("Not found")
    elif response.status_code == 403:
        reset_timestamp = int(response.headers.get('X-RateLimit-Reset'))
        reset_time = datetime.fromtimestamp(reset_timestamp)
        minutes_until_reset = (reset_time -
                               datetime.now()).total_seconds() // 60
        print(f"Reset in {int(minutes_until_reset)} min")
    elif response.status_code == 200:
        location = response.json().get('location', 'Not found')
        print(location)
    else:
        print("Failed to fetch data")


if __name__ == '__main__':
    if len(sys.argv) > 1:
        url = sys.argv[1]
        get_user_location(url)
    else:
        print("Usage: ./2-user_location.py <url>")
