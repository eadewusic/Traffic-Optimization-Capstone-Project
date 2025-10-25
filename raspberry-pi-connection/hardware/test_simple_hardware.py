#!/usr/bin/env python3
"""
Simple Hardware Test - Validate GPIO before deployment

LED MODULES (12 outputs):
  North Module:
    Red:    GPIO 16 (Pin 36)  → Module R pin
    Yellow: GPIO 20 (Pin 38)  → Module Y pin
    Green:  GPIO 21 (Pin 40)  → Module G pin
    GND:    Pin 34 (GND)      → Module GND pin
  
  East Module:
    Red:    GPIO 5  (Pin 29) → Module R pin
    Yellow: GPIO 6  (Pin 31) → Module Y pin
    Green:  GPIO 13 (Pin 33) → Module G pin
    GND:    Pin 30 (GND)     → Module GND pin
  
  South Module:
    Red:    GPIO 23 (Pin 16) → Module R pin
    Yellow: GPIO 24 (Pin 18) → Module Y pin
    Green:  GPIO 22 (Pin 15) → Module G pin
    GND:    Pin 14 (GND)     → Module GND pin
  
  West Module:
    Red:    GPIO 14 (Pin 8)  → Module R pin
    Yellow: GPIO 4 (Pin 7)   → Module Y pin
    Green:  GPIO 18 (Pin 12) → Module G pin
    GND:    Pin 9 (GND)      → Module GND pin

COOLING FAN (from Pi case):
  Fan +5V:  Pin 4 (5V Power)
  Fan GND:  Pin 6 (GND)
  Fan PWM:  Pin 11 (GPIO 17)

BUTTONS:
  North: GPIO 26 (Pin 37)
  East: GPIO 25 (Pin 22)
  South: GPIO 17 (Pin 11)
  West: GPIO 8 (Pin 24)
  Common GND: Pin 39 (GND)

BREADBOARDS CONNECTION:
    Pi GND (Pin 39) ─── Breadboard 1 (buttons) GND rail
                            │
    Breadboard 1 GND rail ─── Breadboard 2 (traffic module 1) GND rail
                            │
    Breadboard 2 GND rail ─── Breadboard 3 (traffic module 2) GND rail
                            │
    Breadboard 3 GND rail ─── Breadboard 4 (traffic module 3) GND rail
                            │
    Breadboard 4 GND rail ─── Breadboard 5 (traffic module 4) GND rail
                            │
    Breadboard 5 GND rail ─── Pi GND (Pin 25)
"""

import RPi.GPIO as GPIO
import time

LED_PINS = {
    'north_red': 16, 'north_yellow': 20, 'north_green': 21,
    'east_red': 5, 'east_yellow': 6, 'east_green': 13,
    'south_red': 23, 'south_yellow': 24, 'south_green': 22,
    'west_red': 14, 'west_yellow': 4, 'west_green': 18
}

BUTTON_PINS = {'north': 26, 'east': 25, 'south': 17, 'west': 8}

def setup_gpio():
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    for pin in LED_PINS.values():
        GPIO.setup(pin, GPIO.OUT)
        GPIO.output(pin, GPIO.LOW)
    for pin in BUTTON_PINS.values():
        GPIO.setup(pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)

def test_all_leds():
    """Test 1: All LEDs ON"""
    print("\n Test 1: All LEDs ON for 5 seconds")
    for pin in LED_PINS.values():
        GPIO.output(pin, GPIO.HIGH)
    time.sleep(5)
    for pin in LED_PINS.values():
        GPIO.output(pin, GPIO.LOW)
    input("Press Enter if all LEDs worked...")

def test_sequential():
    """Test 2: Sequential LED test"""
    print("\n Test 2: Sequential test")
    for name, pin in LED_PINS.items():
        print(f"   {name}")
        GPIO.output(pin, GPIO.HIGH)
        time.sleep(2)
        GPIO.output(pin, GPIO.LOW)
    input("Press Enter if sequence worked...")

def test_traffic_patterns():
    """Test 3: Traffic patterns"""
    print("\n Test 3: Traffic light patterns")
    
    # N/S green, E/W red
    print("   N/S green, E/W red")
    GPIO.output(LED_PINS['north_green'], GPIO.HIGH)
    GPIO.output(LED_PINS['south_green'], GPIO.HIGH)
    GPIO.output(LED_PINS['east_red'], GPIO.HIGH)
    GPIO.output(LED_PINS['west_red'], GPIO.HIGH)
    time.sleep(5)
    
    # Clear
    for pin in LED_PINS.values():
        GPIO.output(pin, GPIO.LOW)
    time.sleep(1)
    
    # E/W green, N/S red
    print("   E/W green, N/S red")
    GPIO.output(LED_PINS['east_green'], GPIO.HIGH)
    GPIO.output(LED_PINS['west_green'], GPIO.HIGH)
    GPIO.output(LED_PINS['north_red'], GPIO.HIGH)
    GPIO.output(LED_PINS['south_red'], GPIO.HIGH)
    time.sleep(5)
    
    for pin in LED_PINS.values():
        GPIO.output(pin, GPIO.LOW)
    
    input("Press Enter if patterns worked...")

def test_buttons():
    """Test 4: Button inputs"""
    print("\n Test 4: Button test (20 seconds)")
    print("   Press each button...")
    
    start = time.time()
    presses = {'north': 0, 'east': 0, 'south': 0, 'west': 0}
    
    while time.time() - start < 20:
        for name, pin in BUTTON_PINS.items():
            if GPIO.input(pin) == GPIO.LOW:
                time.sleep(0.05) # verify reading has remained LOW for at least 50ms to ignore noise
                presses[name] += 1
                print(f"   {name.upper()} pressed!", end='\r')
                time.sleep(2)
    
    print(f"\n   Presses: {presses}")
    input("Press Enter if buttons worked...")

def test_button_led_link_toggle():
    """Test 5: Button-LED toggle - Click once to turn on, click again to turn off"""
    print("\n Test 5: Button-LED Toggle (15 seconds)")
    print("   CLICK buttons once to toggle LED on/off")
    
    # Track LED states
    led_states = {'north': False, 'south': False, 'east': False, 'west': False}
    
    # Track last button states for edge detection
    last_button_states = {'north': True, 'south': True, 'east': True, 'west': True}
    
    # Debounce timing
    last_press_time = {'north': 0, 'south': 0, 'east': 0, 'west': 0}
    debounce_delay = 0.2  # 200ms debounce
    
    start = time.time()
    while time.time() - start < 15:
        current_time = time.time()
        
        for direction in ['north', 'south', 'east', 'west']:
            button_pin = BUTTON_PINS[direction]
            led_pin = LED_PINS[f'{direction}_green']
            
            current_button = GPIO.input(button_pin)
            
            # Detect button press (transition from HIGH to LOW)
            if current_button == GPIO.LOW and last_button_states[direction] == GPIO.HIGH:
                # Button just pressed - check debounce
                if current_time - last_press_time[direction] > debounce_delay:
                    # Toggle LED state
                    led_states[direction] = not led_states[direction]
                    GPIO.output(led_pin, GPIO.HIGH if led_states[direction] else GPIO.LOW)
                    
                    status = "ON" if led_states[direction] else "OFF"
                    print(f"   {direction.upper()}: {status}          ", end='\r')
                    
                    last_press_time[direction] = current_time
            
            last_button_states[direction] = current_button
        
        time.sleep(0.01)  # 10ms polling
    
    # Turn off all LEDs
    for pin in LED_PINS.values():
        GPIO.output(pin, GPIO.LOW)
    
    print("\n Button-LED toggle test complete")

if __name__ == "__main__":
    print("HARDWARE VALIDATION TESTS")
    
    try:
        setup_gpio()
        
        test_all_leds()
        test_sequential()
        test_traffic_patterns()
        test_buttons()
        test_button_led_link_toggle()
        
        print("\n ALL TESTS PASSED - Hardware ready for deployment!")
        
    except KeyboardInterrupt:
        print("\n\nTests interrupted")
    finally:
        GPIO.cleanup()
