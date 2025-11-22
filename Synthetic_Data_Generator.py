
import os
import pandas as pd
import json
import numpy as np


# ------------------------------------------------
#  Function: Read baseline voltage from sample BMS file
# ------------------------------------------------
def read_baseline_voltage(bms_path):
    """Extract initial pack voltage from sample BMS CSV."""
    if not os.path.exists(bms_path):
        raise FileNotFoundError(
            f"Sample BMS file not found:\n{bms_path}\n\n"
            f"Make sure the file exists and the path is correct."
        )

    df = pd.read_csv(bms_path)
    
    # Get the soc string from the first row
    soc_str = df.loc[0, "soc"]
    
    # Method 1: Try using ast.literal_eval (handles Python literals with single quotes)
    try:
        import ast
        soc_dict = ast.literal_eval(soc_str)
        baseline = float(soc_dict["total_voltage"])
        return baseline
    except (ValueError, SyntaxError, ImportError):
        pass
    
    # Method 2: If ast fails, try simple string replacement
    try:
        json_str = soc_str.replace("'", '"')
        soc_dict = json.loads(json_str)
        baseline = float(soc_dict["total_voltage"])
        return baseline
    except json.JSONDecodeError:
        pass
    
    # Method 3: Last resort - manual parsing
    try:
        # Extract the total_voltage value using string operations
        import re
        match = re.search(r"'total_voltage':\s*([0-9.]+)", soc_str)
        if match:
            baseline = float(match.group(1))
            return baseline
    except:
        pass
    
    raise ValueError(f"Could not parse voltage from: {soc_str}")


# ------------------------------------------------
#  EIS Generator Class
# ------------------------------------------------
class EISGenerator:
    def __init__(self, nominal_voltage, excitation_amplitude=0.5, sampling_rate=50000):
        self.nominal_voltage = float(nominal_voltage)
        self.excitation_amplitude = float(excitation_amplitude)
        self.fs = int(sampling_rate)

        # Realistic Li-ion battery equivalent circuit parameters
        # Model: R_series + (R_ct || C_dl)
        self.R_series = 0.050  # 50 mΩ - series/ohmic resistance
        self.R_ct = 0.030      # 30 mΩ - charge transfer resistance
        self.C_dl = 1.0        # 1 F - double layer capacitance

    # -------------------------
    def calculate_impedance(self, freq_hz):
        """Calculate complex impedance at given frequency using equivalent circuit model."""
        omega = 2 * np.pi * freq_hz
        
        # Parallel RC impedance: Z_ct || C_dl
        # Z = R_ct / (1 + jωR_ct*C_dl)
        Z_ct = self.R_ct / (1 + 1j * omega * self.R_ct * self.C_dl)
        
        # Total impedance: R_series + (R_ct || C_dl)
        Z_total = self.R_series + Z_ct
        
        return Z_total

    # -------------------------
    def generate(self, freq_hz, duration_s):
        """
        Generate synthetic EIS data.
        
        Returns:
            t: time array
            voltage: voltage response array
            current: current excitation array
            mag: impedance magnitude
            phase: impedance phase in radians
        """
        num = int(self.fs * duration_s)
        t = np.linspace(0, duration_s, num, endpoint=False)

        # Sinusoidal current excitation (clean single frequency)
        current = self.excitation_amplitude * np.sin(2 * np.pi * freq_hz * t)

        # Calculate impedance at this frequency
        Z_total = self.calculate_impedance(freq_hz)
        mag = np.abs(Z_total)
        phase = np.angle(Z_total)

        # Generate voltage response
        voltage = self._voltage_response(current, freq_hz, t, Z_total)
        
        return t, voltage, current, mag, phase

    # -------------------------
    def _voltage_response(self, current, freq, t, Z_complex):
        """
        Generate realistic voltage response based on complex impedance.
        
        The voltage response follows:
        V(t) = V_baseline + |Z| * I_amp * sin(ωt + φ)
        where φ is the impedance phase angle.
        """
        # Extract magnitude and phase
        mag = np.abs(Z_complex)
        phase = np.angle(Z_complex)
        
        # Voltage response with correct phase shift
        # V = V_dc + |Z| * I_amp * sin(ωt + φ)
        voltage_ac = mag * self.excitation_amplitude * np.sin(
            2 * np.pi * freq * t + phase
        )
        
        # Add realistic measurement noise and small drift
        noise = 0.001 * np.random.normal(size=len(t))  # 1 mV RMS noise
        drift = 0.0005 * np.sin(2 * np.pi * 0.1 * t)   # Slow thermal drift
        
        voltage = self.nominal_voltage + voltage_ac + noise + drift
        
        return voltage

    # -------------------------
    def to_bms_dataframe(self, t, voltage, current, downsample_factor=1):
        """
        Convert time-series data to BMS dataframe format.
        
        Args:
            t: time array
            voltage: voltage array
            current: current array
            downsample_factor: keep every Nth sample (1 = keep all)
        """
        data = []
        n = len(t)
        
        # Downsample if requested
        indices = range(0, n, downsample_factor)

        for i in indices:
            # Distribute pack voltage across 8 cells with small variations
            base_cell = voltage[i] / 8.0
            cell_voltages = {
                k: float(base_cell + 0.001 * np.sin(2 * np.pi * 0.05 * t[i] + k * 0.26))
                for k in range(1, 9)
            }

            vals = list(cell_voltages.values())
            hi = max(vals)
            lo = min(vals)
            hi_i = vals.index(hi) + 1
            lo_i = vals.index(lo) + 1

            soc_obj = {
                "total_voltage": float(np.round(voltage[i], 6)),
                "current": float(np.round(current[i], 6)),
                "soc_percent": float(50.0 + 0.1 * np.sin(2 * np.pi * 0.01 * t[i])),
            }

            row = {
                "soc": json.dumps(soc_obj),
                "cell_voltage_range": json.dumps(
                    {
                        "highest_voltage": float(np.round(hi, 6)),
                        "highest_cell": hi_i,
                        "lowest_voltage": float(np.round(lo, 6)),
                        "lowest_cell": lo_i,
                    }
                ),
                "temperature_range": json.dumps(
                    {
                        "highest_temperature": 18,
                        "highest_sensor": 1,
                        "lowest_temperature": 18,
                        "lowest_sensor": 1,
                    }
                ),
                "mosfet_status": json.dumps(
                    {
                        "mode": "stationary",
                        "charging_mosfet": True,
                        "discharging_mosfet": True,
                        "capacity_ah": 14.85,
                    }
                ),
                "status": json.dumps(
                    {
                        "cells": 8,
                        "temperature_sensors": 2,
                        "charger_running": False,
                        "load_running": False,
                        "states": {"DI1": False},
                        "cycles": 0,
                    }
                ),
                "cell_voltages": json.dumps(cell_voltages),
                "temperatures": json.dumps({1: 18, 2: 18}),
                "errors": json.dumps([]),
            }
            data.append(row)

        return pd.DataFrame(data)

    # -------------------------
    def plot(self, t, voltage, current, freq, out_csv):
        """Generate comprehensive waveform plots."""
        # Show first 10 cycles or up to 100ms, whichever is less
        cycles = min(10, int(freq * t[-1]))
        samples = min(len(t), int(cycles * self.fs / freq))
        samples = max(samples, 100)  # Show at least 100 samples

        fig, axs = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
        
        # Plot 1: Current excitation
        axs[0].plot(t[:samples] * 1000, current[:samples], 'b-', linewidth=1.5)
        axs[0].set_ylabel("Current (A)", fontsize=11)
        axs[0].set_title(f"EIS Excitation Signal - {freq} Hz", fontsize=12, fontweight='bold')
        axs[0].grid(True, alpha=0.3)
        axs[0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
        
        # Plot 2: Voltage response
        axs[1].plot(t[:samples] * 1000, voltage[:samples], 'r-', linewidth=1.5)
        axs[1].set_ylabel("Voltage (V)", fontsize=11)
        axs[1].grid(True, alpha=0.3)
        axs[1].axhline(y=self.nominal_voltage, color='k', linestyle='--', alpha=0.3, label='Baseline')
        
        # Plot 3: Normalized overlay to show phase relationship
        curr_norm = current[:samples] / (np.max(np.abs(current[:samples])) + 1e-12)
        volt_ac = voltage[:samples] - self.nominal_voltage
        volt_norm = volt_ac / (np.max(np.abs(volt_ac)) + 1e-12)
        
        axs[2].plot(t[:samples] * 1000, curr_norm, 'b-', label='Current (norm)', linewidth=1.5)
        axs[2].plot(t[:samples] * 1000, volt_norm, 'r-', label='Voltage (norm)', linewidth=1.5, alpha=0.8)
        axs[2].legend(loc='upper right', fontsize=10)
        axs[2].set_ylabel("Normalized", fontsize=11)
        axs[2].set_xlabel("Time (ms)", fontsize=11)
        axs[2].grid(True, alpha=0.3)
        axs[2].axhline(y=0, color='k', linestyle='--', alpha=0.3)

        png_path = out_csv.replace(".csv", "_waveforms.png")
        plt.tight_layout()
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        plt.close()
        return png_path


# ------------------------------------------------
#  MAIN
# ------------------------------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load baseline voltage automatically
    baseline_voltage = read_baseline_voltage(SAMPLE_BMS_PATH)
    print(f"Baseline voltage extracted: {baseline_voltage} V")

    # Initialize generator
    gen = EISGenerator(
        nominal_voltage=baseline_voltage,
        excitation_amplitude=EXCITATION_AMP,
        sampling_rate=SAMPLING_RATE,
    )

    produced = []
    
    print("\n" + "="*70)
    print("EIS SYNTHETIC DATA GENERATOR - HIGH RESOLUTION MODE")
    print("="*70)

    for f in FREQUENCIES:
        # Set duration based on frequency to ensure:
        # - Good FFT resolution (many frequency bins)
        # - Sufficient cycles for stable measurement
        # - Reasonable file sizes
        if f <= 100:
            duration = 1.0      # 1 second = 100 cycles at 100 Hz
        elif f <= 1000:
            duration = 0.5      # 0.5 second = 500 cycles at 1000 Hz
        else:
            duration = 0.1      # 0.1 second = 500 cycles at 5000 Hz
        
        num_samples = int(duration * SAMPLING_RATE)
        num_cycles = int(duration * f)
        
        print(f"\n{'─'*70}")
        print(f"Generating {f} Hz EIS Data")
        print(f"{'─'*70}")
        print(f"Duration:      {duration:.3f} s")
        print(f"Cycles:        {num_cycles}")
        print(f"Samples:       {num_samples:,}")
        print(f"Sampling Rate: {SAMPLING_RATE:,} Hz")

        # Generate high-resolution data
        t, v, i, mag, phase = gen.generate(f, duration)
        
        # Convert to BMS format - keep ALL samples (no downsampling)
        print(f"\nConverting to BMS format...")
        df_bms = gen.to_bms_dataframe(t, v, i, downsample_factor=1)

        # Save CSV
        out_csv = os.path.join(OUTPUT_DIR, f"eis_synthetic_{f}hz.csv")
        df_bms.to_csv(out_csv, index=False)
        
        file_size_mb = os.path.getsize(out_csv) / (1024 * 1024)

        # Generate plots
        png = gen.plot(t, v, i, f, out_csv)

        print(f"\n✓ CSV saved:   {out_csv}")
        print(f"  Rows:        {len(df_bms):,}")
        print(f"  File size:   {file_size_mb:.2f} MB")
        print(f"\n✓ Plot saved:  {png}")
        print(f"\n📊 Expected Impedance:")
        print(f"  |Z| = {mag*1000:.3f} mΩ")
        print(f"  ∠Z  = {np.degrees(phase):.2f}°")
        
        produced.append(out_csv)

    print("\n" + "="*70)
    print("✓ All files generated successfully!")
    print("="*70)
    print("\n📋 Expected Results Summary:")
    print("┌─────────────┬──────────────┬─────────────┐")
    print("│  Frequency  │  Impedance   │    Phase    │")
    print("├─────────────┼──────────────┼─────────────┤")
    
    for f in FREQUENCIES:
        Z = gen.calculate_impedance(f)
        mag = np.abs(Z) * 1000  # Convert to mΩ
        phase = np.degrees(np.angle(Z))
        print(f"│   {f:>5} Hz  │  {mag:>6.2f} mΩ  │  {phase:>6.2f}°  │")
    
    print("└─────────────┴──────────────┴─────────────┘")
    print("\n💡 Tips:")
    print("  • Impedance magnitude should decrease slightly with frequency")
    print("  • Phase should be negative (capacitive behavior)")
    print("  • Values should be in 50-70 mΩ range for typical Li-ion")
    print("\n▶ Next step: Run Impedance_Calculator.py to verify results")


# ------------------------------------------------
if __name__ == "__main__":
    main()