import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq
import os
import joblib

# Constants matching Synthetic_Data_Generator.py
FREQUENCIES = [100, 1000, 5000]
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = BASE_DIR
OUTPUT_DIR = BASE_DIR
MODEL_PATH = os.path.join(BASE_DIR, "soh_model.pkl")

class EISAnalyzer:
    def __init__(self, sampling_rate=50000):
        self.fs = sampling_rate
        
    def load_eis_data(self, csv_path):
        """Load EIS data from BMS-formatted CSV file"""
        df = pd.read_csv(csv_path)
        
        # Extract time series data
        times = []
        voltages = []
        currents = []
        
        for idx, row in df.iterrows():
            soc_data = json.loads(row['soc'])
            voltages.append(soc_data['total_voltage'])
            currents.append(soc_data['current'])
            times.append(idx / self.fs)  # Assuming constant sampling rate
            
        return np.array(times), np.array(voltages), np.array(currents)
    
    def compute_impedance_fft(self, times, voltage, current, excitation_freq):
        """
        Compute complex impedance using FFT at excitation frequency.
        
        CRITICAL: Do NOT align or shift signals before FFT.
        The phase information in the FFT directly gives us the impedance phase.
        """
        # Remove DC component (mean)
        voltage_ac = voltage - np.mean(voltage)
        current_ac = current - np.mean(current)
        
        # Apply windowing to reduce spectral leakage
        window = signal.windows.blackmanharris(len(voltage_ac))
        v_windowed = voltage_ac * window
        i_windowed = current_ac * window
        
        # Compute FFT
        v_fft = fft(v_windowed)
        i_fft = fft(i_windowed)
        
        # Compute frequencies
        freqs = fftfreq(len(voltage_ac), 1/self.fs)
        
        # Find index closest to excitation frequency
        target_idx = np.argmin(np.abs(freqs - excitation_freq))
        target_freq = freqs[target_idx]
        
        # Extract complex values at excitation frequency
        v_complex = v_fft[target_idx]
        i_complex = i_fft[target_idx]
        
        # Compute impedance: Z = V / I
        impedance_complex = v_complex / i_complex
        
        # Calculate magnitude and phase
        impedance_magnitude = np.abs(impedance_complex)
        impedance_phase = np.angle(impedance_complex)
        
        print(f"Excitation frequency: {excitation_freq} Hz")
        print(f"Nearest FFT bin: {target_freq:.2f} Hz")
        print(f"Impedance magnitude: {impedance_magnitude:.6f} Ohm ({impedance_magnitude*1000:.3f} mOhm)")
        print(f"Impedance phase: {impedance_phase:.6f} rad ({np.degrees(impedance_phase):.2f} deg)")
        
        # Frequency bin accuracy check
        freq_error = abs(target_freq - excitation_freq)
        if freq_error > excitation_freq * 0.01:  # More than 1% error
            print(f"[WARN] Warning: Frequency bin mismatch of {freq_error:.2f} Hz ({freq_error/excitation_freq*100:.1f}%)")
        
        return impedance_complex, impedance_magnitude, impedance_phase, target_freq
    
    def plot_impedance_analysis(self, times, voltage, current, 
                              excitation_freq, impedance_complex, out_dir):
        """Generate comprehensive analysis plots"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Time domain signals (first few cycles)
        cycles_to_show = min(10, int(excitation_freq * times[-1]))
        samples_to_show = min(len(times), int(cycles_to_show * self.fs / excitation_freq))
        samples_to_show = max(samples_to_show, 100)  # At least 100 samples
        
        # Plot 1: Original signals
        axes[0,0].plot(times[:samples_to_show] * 1000, voltage[:samples_to_show], 
                      label='Voltage', linewidth=2, color='red')
        ax_twin = axes[0,0].twinx()
        ax_twin.plot(times[:samples_to_show] * 1000, current[:samples_to_show], 
                    label='Current', linewidth=2, alpha=0.7, color='blue')
        axes[0,0].set_xlabel('Time (ms)')
        axes[0,0].set_ylabel('Voltage (V)', color='red')
        ax_twin.set_ylabel('Current (A)', color='blue')
        axes[0,0].set_title('Time Domain Signals')
        axes[0,0].grid(True, alpha=0.3)
        
        # Plot 2: Normalized comparison to see phase relationship
        v_ac = voltage[:samples_to_show] - np.mean(voltage[:samples_to_show])
        i_ac = current[:samples_to_show] - np.mean(current[:samples_to_show])
        
        v_norm = v_ac / (np.max(np.abs(v_ac)) + 1e-12)
        i_norm = i_ac / (np.max(np.abs(i_ac)) + 1e-12)
        
        axes[0,1].plot(times[:samples_to_show] * 1000, v_norm, 
                      label='Voltage (norm)', linewidth=2, color='red')
        axes[0,1].plot(times[:samples_to_show] * 1000, i_norm, 
                      label='Current (norm)', linewidth=2, alpha=0.7, color='blue')
        axes[0,1].set_xlabel('Time (ms)')
        axes[0,1].set_ylabel('Normalized Amplitude')
        axes[0,1].set_title('Phase Relationship')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        axes[0,1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
        
        # Plot 3: V-I Lissajous plot
        axes[0,2].plot(current[:samples_to_show], voltage[:samples_to_show], 
                      'b-', linewidth=1.5, alpha=0.6)
        axes[0,2].set_xlabel('Current (A)')
        axes[0,2].set_ylabel('Voltage (V)')
        axes[0,2].set_title('V-I Characteristic')
        axes[0,2].grid(True, alpha=0.3)
        
        # Plot 4: Frequency spectrum - Voltage
        v_ac_full = voltage - np.mean(voltage)
        window = signal.windows.blackmanharris(len(v_ac_full))
        v_fft = fft(v_ac_full * window)
        v_freqs = fftfreq(len(v_ac_full), 1/self.fs)
        positive_freqs = v_freqs[:len(v_freqs)//2]
        v_spectrum = np.abs(v_fft[:len(v_fft)//2])
        
        axes[1,0].semilogy(positive_freqs, v_spectrum, 'r-', linewidth=1)
        axes[1,0].axvline(x=excitation_freq, color='g', linestyle='--', linewidth=2,
                         label=f'Excitation: {excitation_freq} Hz')
        axes[1,0].set_xlabel('Frequency (Hz)')
        axes[1,0].set_ylabel('Magnitude (V)')
        axes[1,0].set_title('Voltage Frequency Spectrum')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        axes[1,0].set_xlim(0, min(excitation_freq * 3, self.fs/2))
        
        # Plot 5: Frequency spectrum - Current
        i_ac_full = current - np.mean(current)
        i_fft = fft(i_ac_full * window)
        i_spectrum = np.abs(i_fft[:len(i_fft)//2])
        
        axes[1,1].semilogy(positive_freqs, i_spectrum, 'b-', linewidth=1)
        axes[1,1].axvline(x=excitation_freq, color='g', linestyle='--', linewidth=2,
                         label=f'Excitation: {excitation_freq} Hz')
        axes[1,1].set_xlabel('Frequency (Hz)')
        axes[1,1].set_ylabel('Magnitude (A)')
        axes[1,1].set_title('Current Frequency Spectrum')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        axes[1,1].set_xlim(0, min(excitation_freq * 3, self.fs/2))
        
        # Plot 6: Nyquist plot
        axes[1,2].plot(np.real(impedance_complex), -np.imag(impedance_complex), 
                      'ro', markersize=12, markeredgewidth=2, markeredgecolor='darkred',
                      markerfacecolor='red')
        axes[1,2].set_xlabel('Real(Z) (Ohm)')
        axes[1,2].set_ylabel('-Imag(Z) (Ohm)')
        axes[1,2].set_title(f'Impedance at {excitation_freq} Hz')
        axes[1,2].grid(True, alpha=0.3)
        axes[1,2].axhline(y=0, color='k', linestyle='-', alpha=0.3)
        axes[1,2].axvline(x=0, color='k', linestyle='-', alpha=0.3)
        
        # Add annotation box
        Z_real = np.real(impedance_complex) * 1000  # mOhm
        Z_imag = np.imag(impedance_complex) * 1000  # mOhm
        Z_mag = np.abs(impedance_complex) * 1000    # mOhm
        Z_phase = np.degrees(np.angle(impedance_complex))
        
        annotation_text = (
            f'|Z| = {Z_mag:.3f} mOhm\n'
            f'Ang = {Z_phase:.2f} deg\n'
            f'Re(Z) = {Z_real:.3f} mOhm\n'
            f'Im(Z) = {Z_imag:.3f} mOhm'
        )
        
        axes[1,2].text(0.05, 0.95, annotation_text,
                      transform=axes[1,2].transAxes,
                      fontsize=10,
                      verticalalignment='top',
                      bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
def main():
    analyzer = EISAnalyzer()
    results = []

    print("EIS IMPEDANCE ANALYSIS")
    print("="*70)
    
    # Load SOH Model
    try:
        soh_model = joblib.load(MODEL_PATH)
        print(f"[INFO] Loaded SOH Model from {MODEL_PATH}")
    except Exception as e:
        print(f"[WARN] Could not load SOH model: {e}")
        soh_model = None
        
    # Dictionary to store features for SOH prediction
    soh_features = {}
    
    for freq in FREQUENCIES:
        print(f"\n{'-'*70}")
        print(f"Processing {freq} Hz EIS data")
        print(f"{'-'*70}")
        
        # Load data
        csv_path = os.path.join(INPUT_DIR, f"eis_synthetic_{freq}hz.csv")
        
        if not os.path.exists(csv_path):
            print(f"[ERR] ERROR: File not found: {csv_path}")
            continue
            
        times, voltages, currents = analyzer.load_eis_data(csv_path)
        
        print(f"Loaded {len(times)} samples ({times[-1]:.3f} seconds)")
        print(f"Sampling rate: {1/(times[1]-times[0]):.1f} Hz")
        
        # Compute impedance using FFT (no alignment needed)
        impedance_complex, magnitude, phase, actual_freq = analyzer.compute_impedance_fft(
            times, voltages, currents, freq
        )
        
        # Generate analysis plots
        plot_path = analyzer.plot_impedance_analysis(
            times, voltages, currents,
            freq, impedance_complex, OUTPUT_DIR
        )
        
        # Store results
        results.append({
            'frequency_hz': freq,
            'actual_frequency_hz': actual_freq,
            'impedance_magnitude_ohm': magnitude,
            'impedance_magnitude_mohm': magnitude * 1000,
            'impedance_phase_rad': phase,
            'impedance_phase_deg': np.degrees(phase),
            'impedance_real_ohm': np.real(impedance_complex),
            'impedance_imag_ohm': np.imag(impedance_complex),
            'impedance_real_mohm': np.real(impedance_complex) * 1000,
            'impedance_imag_mohm': np.imag(impedance_complex) * 1000,
            'plot_path': plot_path
        })
        
        # Collect features for SOH prediction
        soh_features[f'Z_mag_{freq}Hz'] = magnitude
        soh_features[f'Z_phase_{freq}Hz'] = phase
        soh_features[f'Z_real_{freq}Hz'] = np.real(impedance_complex)
        soh_features[f'Z_imag_{freq}Hz'] = np.imag(impedance_complex)
        
        print(f"[OK] Analysis complete")
        print(f"[OK] Plot saved: {plot_path}")
    
    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_csv = os.path.join(OUTPUT_DIR, "impedance_results_summary.csv")
    results_df.to_csv(results_csv, index=False)
    
    print(f"\n{'='*70}")
    print("[OK] EIS Analysis Complete!")
    print(f"{'='*70}")
    print(f"Results summary saved: {results_csv}")
    
    # Print final results table
    print("\n" + "="*70)
    print("IMPEDANCE RESULTS SUMMARY")
    print("="*70)
    print("\n+-------------+--------------+-------------+--------------+")
    print("|  Frequency  |  |Z| (mOhm)  |   Ang (deg) |  Re(Z) (mOhm)|")
    print("+-------------+--------------+-------------+--------------+")
    
    for _, row in results_df.iterrows():
        freq = row['frequency_hz']
        mag = row['impedance_magnitude_mohm']
        phase = row['impedance_phase_deg']
        real = row['impedance_real_mohm']
        print(f"|   {freq:>5} Hz  |   {mag:>7.3f}    |  {phase:>7.2f}    |   {real:>7.3f}    |")
    
    print("+-------------+--------------+-------------+--------------+")
    
    # Validation check
    print("\n" + "="*70)
    print("VALIDATION")
    print("="*70)
    for _, row in results_df.iterrows():
        freq = row['frequency_hz']
        mag = row['impedance_magnitude_mohm']
        phase = row['impedance_phase_deg']
        
        # Check if results are reasonable
        mag_ok = 40 <= mag <= 90  # Reasonable range for Li-ion
        phase_ok = -45 <= phase <= 5  # Capacitive behavior expected
        
        status = "[OK]" if (mag_ok and phase_ok) else "[WARN]"
        print(f"{status} {freq:>5} Hz: |Z|={mag:.3f} mOhm, Ang={phase:.2f} deg")
    
    print("\n[INFO] Expected trends:")
    print("  * Impedance magnitude: ~50 mOhm, decreasing slightly with frequency")
    print("  * Phase angle: negative (capacitive), decreasing magnitude with frequency")
    
    # SOH Prediction
    if soh_model is not None and len(soh_features) == len(FREQUENCIES) * 4:
        print("\n" + "="*70)
        print("STATE OF HEALTH (SOH) ESTIMATION")
        print("="*70)
        
        # Create DataFrame for prediction (single row)
        X_pred = pd.DataFrame([soh_features])
        
        # Predict
        soh_pred = soh_model.predict(X_pred)[0]
        soh_percent = soh_pred * 100
        
        print(f"Estimated SOH: {soh_pred:.4f} ({soh_percent:.2f}%)")
        
        if soh_percent > 90:
            print("Status: EXCELLENT")
        elif soh_percent > 80:
            print("Status: GOOD")
        elif soh_percent > 70:
            print("Status: FAIR")
        else:
            print("Status: REPLACE BATTERY")
            
    else:
        print("\n[WARN] Skipping SOH prediction (Model not loaded or missing features)")

if __name__ == "__main__":
    main()