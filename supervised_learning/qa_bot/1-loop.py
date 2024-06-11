#!/usr/bin/env python3
"""cript that takes in input from the user with the prompt Q: and prints
A: as a response. If the user inputs exit, quit, goodbye, or bye,
case insensitive, print A: Goodbye and exit."""

while True:
    Q = input("Q: ").lower()
    if Q in {"exit", "quit", "goodbye", "bye"}:
        print("A: Goodbye")
        break
    print("A: ")
