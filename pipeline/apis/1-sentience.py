#!/usr/bin/env python3
""" APIs """
import requests


def sentientPlanets():
    """ returns the list of names of home planets of all sentient species """
    url = 'https://swapi-api.hbtn.io/api/species/'
    planets = set()

    while url:
        response = requests.get(url)
        data = response.json()

        for species in data['results']:
            classification = species.get('classification', '').lower()
            designation = species.get('designation', '').lower()
            if 'sentient' in classification or 'sentient' in designation:
                homeworld_url = species.get('homeworld')
                if homeworld_url:
                    homeworld_response = requests.get(homeworld_url)
                    homeworld_data = homeworld_response.json()
                    planets.add(homeworld_data.get('name', 'unknown'))

        url = data['next']

    return list(planets)
