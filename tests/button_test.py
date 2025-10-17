#!/usr/bin/env python3
"""
Button Hardware Test Script

Initialize specified GPIO pins as inputs with internal pull-up resistors
and continuously monitor the state of each button. When a button is pressed
(signaling a LOW state), it prints a confirmation message to the console.

Pinout (BCM):
  North: GPIO 26 (Pin 37)
  East: GPIO 25 (Pin 22)
  South: GPIO 27 (Pin 13)
  West: GPIO 8 (Pin 24)
"""

import RPi.GPIO as GPIO
import time

BUTTONS = {
    "North": 26,
    "East": 25,
    "South": 27,
    "West": 8
}

GPIO.setmode(GPIO.BCM)

# Setup all buttons with pull-up resistors
# The pin is an input (GPIO.IN) and the internal resistor is set to pull-up (GPIO.PUD_UP)
# This means the pin reads HIGH normally, and LOW when the button is pressed
for name, pin in BUTTONS.items():
    GPIO.setup(pin, GPIO.IN, pull_up_down=GPIO.PUD_UP) # GPIO.PUD_UP for stable readings

print("Testing all buttons. Press each one to verify input...")
print("Press Ctrl+C to stop.\n")

try:
    while True:
        for name, pin in BUTTONS.items():
            if GPIO.input(pin) == GPIO.LOW:
                print(f"{name} button pressed!")
                time.sleep(0.3)
except KeyboardInterrupt:
    GPIO.cleanup()
    print("Button test complete.")
