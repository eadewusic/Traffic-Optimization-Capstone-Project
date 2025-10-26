#!/usr/bin/env python3
"""
Full Hardware Connection Test Script

Controls four traffic light modules (North, East, South, West)
using a set of connected LEDs and four corresponding push buttons. Pressing a button
activates a traffic cycle for that specific lane, moving from Red to Green, then
Yellow, and back to Red, while all other lanes remain Red.

Hardware Connections:
---------------------
LED MODULES (Output):
- North: Red(GPIO 16), Yellow(GPIO 20), Green(GPIO 21)
- East:  Red(GPIO 5),  Yellow(GPIO 6),  Green(GPIO 13)
- South: Red(GPIO 23), Yellow(GPIO 24), Green(GPIO 25)
- West:  Red(GPIO 14), Yellow(GPIO 4),  Green(GPIO 18)

BUTTONS (Input - connected to GND):
- North: GPIO 9
- East:  GPIO 10
- South: GPIO 22
- West:  GPIO 17

Note: Buttons are configured with internal PULL_UP resistors, meaning the input is
LOW when the button is pressed (connected to GND).

Functions:
- set_all_red(): Sets all traffic light modules to RED.

Execution Flow:
1. Initialize GPIO, set BCM mode.
2. Define LED and Button pin mappings.
3. Setup all LED pins as outputs (initial state LOW).
4. Setup all Button pins as inputs with PULL_UP resistors.
5. Immediately call set_all_red() to begin in a safe state.
6. Enters an infinite loop to poll button inputs.
7. If a button is pressed, the corresponding lane's traffic cycle runs
(Red -> Green -> Yellow -> Red).
8. Handles cleanup of GPIO on KeyboardInterrupt (Ctrl+C).
"""

import RPi.GPIO as GPIO
import time

# LED PINS
LED_MODULES = {
    "North": {"R": 16, "Y": 20, "G": 21},
    "East":  {"R": 5,  "Y": 6,  "G": 13},
    "South": {"R": 23, "Y": 24, "G": 25},
    "West":  {"R": 14, "Y": 4, "G": 18}
}

# BUTTON PINS
BUTTONS = {
    "North": 9,
    "East": 10,
    "South": 22,
    "West": 17
}

GPIO.setmode(GPIO.BCM)

# Setup LEDs
for module in LED_MODULES.values():
    for pin in module.values():
        GPIO.setup(pin, GPIO.OUT)
        GPIO.output(pin, GPIO.LOW)

# Setup Buttons
for pin in BUTTONS.values():
    GPIO.setup(pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)

print("System Ready. Press a button to activate lane...")

def set_all_red():
    for leds in LED_MODULES.values():
        GPIO.output(leds["R"], GPIO.HIGH)
        GPIO.output(leds["Y"], GPIO.LOW)
        GPIO.output(leds["G"], GPIO.LOW)

try:
    set_all_red()
    while True:
        for lane, button in BUTTONS.items():
            if GPIO.input(button) == GPIO.LOW:  # Button pressed
                print(f"{lane} lane activated!")
                set_all_red()
                leds = LED_MODULES[lane]
                GPIO.output(leds["R"], GPIO.LOW)
                GPIO.output(leds["G"], GPIO.HIGH)
                time.sleep(2)  # Light ON duration
                GPIO.output(leds["G"], GPIO.LOW)
                GPIO.output(leds["Y"], GPIO.HIGH)
                time.sleep(1)
                GPIO.output(leds["Y"], GPIO.LOW)
                GPIO.output(leds["R"], GPIO.HIGH)
                print(f"{lane} cycle complete.\n")
        time.sleep(0.1)

except KeyboardInterrupt:
    GPIO.cleanup()
    print("Program ended.")
