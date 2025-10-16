#!/usr/bin/env python3
"""
Traffic Light Hardware Test Script

Tests all 4 LED modules (North, East, South, West) by cycling through
each color (Red, Yellow, Green) sequentially to verify GPIO connections
and LED functionality on Raspberry Pi.

LED MODULES (12 outputs):
  North Module:
    Red:    GPIO 14 (Pin 8)  → Module R pin
    Yellow: GPIO 4 (Pin 7) → Module Y pin
    Green:  GPIO 18 (Pin 12) → Module G pin
    GND:    Pin 9 (GND)      → Module GND pin
  
  East Module:
    Red:    GPIO 23 (Pin 16) → Module R pin
    Yellow: GPIO 24 (Pin 18) → Module Y pin
    Green:  GPIO 22 (Pin 15) → Module G pin
    GND:    Pin 14 (GND)      → Module GND pin
  
  South Module:
    Red:    GPIO 5  (Pin 29) → Module R pin
    Yellow: GPIO 6  (Pin 31) → Module Y pin
    Green:  GPIO 13 (Pin 33) → Module G pin
    GND:    Pin 30 (GND)     → Module GND pin
  
  West Module:
    Red:    GPIO 16 (Pin 36) → Module R pin
    Yellow: GPIO 20 (Pin 38) → Module Y pin
    Green:  GPIO 21 (Pin 40) → Module G pin
    GND:    Pin 34 (GND)     → Module GND pin

COOLING FAN (from Pi case):
  Fan +5V:  Pin 4 (5V Power)
  Fan GND:  Pin 6 (GND)
  Fan PWM:  Pin 11 (GPIO 17)

Usage:
    python3 traffic_test.py
    
Press Ctrl+C to stop and cleanup GPIO pins.
"""

import RPi.GPIO as GPIO
import time

# Define all LED pins for all 4 modules
LED_MODULES = {
    "North": {"R": 14, "Y": 4, "G": 18},
    "East":  {"R": 23, "Y": 24, "G": 22},
    "South": {"R": 5,  "Y": 6,  "G": 13},
    "West":  {"R": 16, "Y": 20, "G": 21}
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
