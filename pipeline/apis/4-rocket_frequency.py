#!/usr/bin/env python3
"""Script to count the number of SpaceX launches per rocket."""
import requests
from collections import defaultdict


def count_launches_per_rocket():
    """Count and display the number of launches per rocket."""
    url = 'https://api.spacexdata.com/v4/launches'
    response = requests.get(url)
    if response.status_code != 200:
        print("Failed to fetch data from SpaceX API")
        return

    launches = response.json()
    if not launches:
        print("No launches found")
        return

    rocket_count = defaultdict(int)

    # Count the launches per rocket
    for launch in launches:
        rocket_id = launch['rocket']
        rocket_count[rocket_id] += 1

    # Get rocket names and map them to counts
    rockets_url = 'https://api.spacexdata.com/v4/rockets'
    rockets_response = requests.get(rockets_url)
    rockets = rockets_response.json()

    rocket_names = {rocket['id']: rocket['name'] for rocket in rockets}
    sorted_rockets = sorted(
        rocket_names.items(),
        key=lambda item: (-rocket_count[item[0]], item[1])
    )

    # Display the results
    for rocket_id, rocket_name in sorted_rockets:
        print("{}: {}".format(rocket_name, rocket_count[rocket_id]))


if __name__ == '__main__':
    count_launches_per_rocket()
