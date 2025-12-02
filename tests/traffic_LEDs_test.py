#!/usr/bin/env python3
"""
Traffic Light Hardware Test Script

Tests all 4 LED modules (North, East, South, West) by cycling through
each color (Red, Yellow, Green) sequentially to verify GPIO connections
and LED functionality on Raspberry Pi.

HARDWARE ARCHITECTURE: 4-Breadboard Modular Design
  - Each breadboard represents one intersection direction
  - Physical cross-pattern layout mirrors actual 4-way intersection
  - One LED module + one button per breadboard
  - Common GND rail connecting all modules to Pi

BREADBOARD 1 (NORTH - Top position):
  LED Module:
    Red:    GPIO 16 (Pin 36)
    Yellow: GPIO 20 (Pin 38)
    Green:  GPIO 21 (Pin 40)
  Button: GPIO 9 (Pin 21)
  
BREADBOARD 2 (EAST - Right position):
  LED Module:
    Red:    GPIO 5  (Pin 29)
    Yellow: GPIO 6  (Pin 31)
    Green:  GPIO 13 (Pin 33)
  Button: GPIO 10 (Pin 19)
  
BREADBOARD 3 (SOUTH - Bottom position):
  LED Module:
    Red:    GPIO 23 (Pin 16)
    Yellow: GPIO 24 (Pin 18)
    Green:  GPIO 25 (Pin 22)
  Button: GPIO 22 (Pin 15)
  
BREADBOARD 4 (WEST - Left position):
  LED Module:
    Red:    GPIO 14 (Pin 8)
    Yellow: GPIO 4  (Pin 7)
    Green:  GPIO 18 (Pin 12)
  Button: GPIO 17 (Pin 11)

RASPBERRY PI 4 Model B 2GB RAM:
  GPIO Connections: 16 pins (4 buttons + 12 LEDs)
  Power: 5V/3A USB-C
  Case: Red/white protective enclosure with cooling fan
  
GROUND CONNECTIONS:
  Common GND rail connects all 4 breadboards to Pi GND pins

COOLING FAN (from Pi case):
  Fan +5V:  Pin 4 (5V Power)
  Fan GND:  Pin 6 (GND)
  Fan PWM:  Pin 5 (GPIO 3)

Usage:
    python3 traffic_test.py
    
Press Ctrl+C to stop and cleanup GPIO pins.
"""

import RPi.GPIO as GPIO
import time

# Define all LED pins for all 4 modules
LED_MODULES = {
    "North": {"R": 16, "Y": 20, "G": 21},
    "East":  {"R": 5,  "Y": 6,  "G": 13},
    "South": {"R": 23, "Y": 24, "G": 25},
    "West":  {"R": 14, "Y": 4, "G": 18}
}

GPIO.setmode(GPIO.BCM)

# Setup all pins as outputs
for module in LED_MODULES.values():
    for pin in module.values():
        GPIO.setup(pin, GPIO.OUT)
        GPIO.output(pin, GPIO.LOW)

print("Testing all traffic light modules...\n")

try:
    while True:
        for module_name, leds in LED_MODULES.items():
            print(f"Testing {module_name} module...")
            for color, pin in leds.items():
                GPIO.output(pin, GPIO.HIGH)
                print(f"  {color} light ON ({module_name})")
                time.sleep(0.7)
                GPIO.output(pin, GPIO.LOW)
            print(f"  {module_name} test complete.\n")
            time.sleep(1)

except KeyboardInterrupt:
    GPIO.cleanup()
    print("Test complete.")
