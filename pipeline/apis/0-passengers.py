#!/bin/bash/python3
""" APIs """
import requests


def availableShips(passengerCount):
    """ create a method that returns the list of ships that
     can hold a given number of passengers """
    url = 'https://swapi-api.hbtn.io/api/starships/'
    them_ships = []

    while url:
        response = requests.get(url)
        data = response.json()

        for ship in data['results']:
            passengers = ship.get('passengers', '0').replace(',', '')
            if passengers.isdigit() and int(passengers) >= passengerCount:
                them_ships.append(ship['name'])

        url = data['next']

    return them_ships

