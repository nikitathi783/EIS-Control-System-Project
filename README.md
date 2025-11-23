# Project EIS (Electrochemical Impedance Spectroscopy)

This project implements a Battery Management System (BMS) utility for Charging, EIS Analysis, and State of Health (SOH) estimation.

## Features

*   **Charging Control**: CC-CV charging profile implementation.
*   **EIS Analysis**: Electrochemical Impedance Spectroscopy analysis using FFT to calculate complex impedance, magnitude, and phase.
*   **SOH Estimation**: Machine Learning model (Random Forest) to predict State of Health based on impedance features.
*   **Synthetic Data Generation**: Tools to generate synthetic EIS waveforms for testing and development.

## Installation

1.  Clone the repository.
2.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Run the main application:

```bash
python main.py
```

Follow the on-screen menu to:
1.  **Charge**: Start the charging process.
2.  **EIS Analysis**: Run the impedance analysis on available data.
3.  **Exit**: Close the application.

## File Structure

*   `main.py`: Entry point of the application.
*   `Defines.py`: Global configuration and constants.
*   `Impedance_Calculator1.py`: Core logic for EIS analysis.
*   `SOH_Estimator.py`: Machine learning model for SOH prediction.
*   `Synthetic_Data_Generator.py`: Generates synthetic test data.
*   `Comm_test.py`: Handles hardware communication.
*   `Ladeskript_1_0.py`: Charging logic implementation.
