#!/usr/bin/env python3
"""
Module for retrieving and displaying information about the first SpaceX launch
using the SpaceX API.
"""
import requests


def get_first_launch():
    """Get first launch with all required details."""
    url = 'https://api.spacexdata.com/v4/launches'
    response = requests.get(url)
    if response.status_code != 200:
        print("Failed to fetch data from SpaceX API")
        return

    launches = response.json()
    if not launches:
        print("No launches found")
        return

    # Sort launches by date_unix
    launches.sort(key=lambda x: x['date_unix'])
    first_launch = launches[0]

    # Get rocket name
    rocket_id = first_launch['rocket']
    rocket_response = requests.get(
        'https://api.spacexdata.com/v4/rockets/{}'.format(rocket_id))
    if rocket_response.status_code == 200:
        rocket_name = rocket_response.json().get('name', "Unknown Rocket")
    else:
        rocket_name = "Unknown Rocket"

    # Get launchpad name & locality
    launchpad_id = first_launch['launchpad']
    launchpad_response = requests.get(
        'https://api.spacexdata.com/v4/launchpads/{}'.format(launchpad_id))
    if launchpad_response.status_code == 200:
        launchpad_data = launchpad_response.json()
        launchpad_name = launchpad_data.get('name', "Unknown Launchpad")
        launchpad_locality = launchpad_data.get('locality', "")
    else:
        launchpad_name = "Unknown Launchpad"
        launchpad_locality = ""

    launch_name = first_launch.get('name', "Unknown Launch")
    launch_date = first_launch.get('date_local', "Unknown Date")

    # Print formatted result
    print("{} ({}) {} - {} ({})".format(
        launch_name, launch_date, rocket_name, launchpad_name,
        launchpad_locality))


if __name__ == '__main__':
    get_first_launch()
