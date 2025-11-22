def setup_square_wave(self, amplitude, offset, frequency, duty_cycle=50):
        """
        Configure square wave function generator
        amplitude: peak-to-peak amplitude in amps
        offset: DC offset in amps
        frequency: frequency in Hz
        duty_cycle: percentage (default 50)
        """
        # Select current mode for function generator
        self.send_command("FUNC:MODE CURR")
        
        # Set static levels (offset and amplitude)
        self.send_command(f"FUNC:SQU:LEV1 {offset}")  # Low level
        self.send_command(f"FUNC:SQU:LEV2 {offset + amplitude}")  # High level
        
        # Set frequency and duty cycle
        self.send_command(f"FUNC:SQU:FREQ {frequency}")
        self.send_command(f"FUNC:SQU:DCYC {duty_cycle}")
        
        # Set rise/fall times to minimum (3μs)
        self.send_command("FUNC:SQU:RISE 0.000003")
        self.send_command("FUNC:SQU:FALL 0.000003")
    
    def start_function_generator(self):
        """Start the function generator"""
        return self.send_command("FUNC ON")
    
    def stop_function_generator(self):
        """Stop the function generator"""
        return self.send_command("FUNC OFF")
    
    
    def test_square_wave(ps, amplitude, offset, frequency, duration):
    """
    Test square wave generation
    amplitude: peak-to-peak amplitude in amps
    offset: DC offset in amps
    frequency: frequency in Hz
    duration: test duration in seconds
    """
    print(f"Starting square wave test: {amplitude}App, {offset}A offset, {frequency}Hz for {duration}s")
    
    # Configure square wave
    ps.setup_square_wave(amplitude, offset, frequency)
    
    # Start function generator
    ps.start_function_generator()
    
    # Monitor for duration
    start_time = time.time()
    while time.time() - start_time < duration:
        # Read actual values
        current = float(ps.send_command("MEAS:CURR?"))
        print(f"Current: {current:.2f}A")
        time.sleep(1/frequency)  # Sample at waveform frequency
    
    # Stop function generator
    ps.stop_function_generator()
    print("Square wave test completed")
    
    if __name__ == "__main__":
    # Initialize power supply controller
    ps = PowerSupplyController(IP_CLIENT2, POWER_SUPPLY_PORT)
    
    if ps.connect():
        try:
            
            # Example square wave test
            test_square_wave(ps, amplitude=2.0, offset=5.0, frequency=1.0, duration=10)
            
        except KeyboardInterrupt:
            print("Stopping...")
        finally:
            # Ensure output is off before disconnecting
            ps.set_output(0)
            ps.close()
    else:
        print("Failed to connect to power supply")