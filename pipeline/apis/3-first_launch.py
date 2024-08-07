#!/usr/bin/env python3
""" APIs """
import requests


def get_first_launch():
    """ get first launch """
    url = 'https://api.spacexdata.com/v4/launches'
    response = requests.get(url)
    if response.status_code != 200:
        print("Failed to fetch data from SpaceX API")
        return

    launches = response.json()
    if not launches:
        print("No upcoming launches found")
        return

    launches.sort(key=lambda x: x['date_unix'])
    first_launch = launches[0]

    # GET rocket name
    rocket_id = first_launch['rocket']
    rocket_response = (
        requests.get(f'https://api.spacexdata.com/v4/rockets/{rocket_id}'))
    rocket_name = rocket_response.json()['name'] \
        if rocket_response.status_code == 200 else "Unknown Rocket"

    # launchpad name & locality
    launchpad_id = first_launch['launchpad']
    launchpad_response = (
        requests.
        get(f'https://api.spacexdata.com/v4/launchpads/{launchpad_id}'))
    if launchpad_response.status_code == 200:
        launchpad_data = launchpad_response.json()
        launchpad_name = launchpad_data['name']
        launchpad_locality = launchpad_data['locality']
    else:
        launchpad_name = "Unknown Launchpad"
        launchpad_locality = ""

    launch_name = first_launch['name']
    launch_date = first_launch['date_local']
    print(f"{launch_name} ({launch_date}) {rocket_name} -"
          f"{launchpad_name} ({launchpad_locality})")


if __name__ == '__main__':
    get_first_launch()
