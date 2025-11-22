# Project EIS

This project implements a system for controlling a Battery Management System (BMS) and Power Supply to perform charging cycles and Electrochemical Impedance Spectroscopy (EIS) analysis.

## Features

*   **Charging Control:** Automated charging cycles with voltage and current monitoring.
*   **EIS Analysis:** Calculates complex impedance, magnitude, and phase from BMS data using FFT.
*   **Visualization:** Generates Nyquist plots, Bode plots, and time-domain signal visualizations.
*   **Synthetic Data Generation:** Simulates Li-ion battery impedance for testing and validation.

## Project Structure

*   `main.py`: Entry point of the application.
*   `Comm_test.py`: Handles network communication with the BMS and Power Supply.
*   `Ladeskript_1_0.py`: Implements the charging logic and SCPI commands.
*   `Impedance_Calculator1.py`: Performs FFT-based impedance analysis.
*   `Synthetic_Data_Generator.py`: Generates synthetic test data.
*   `Defines.py`: Configuration file for IPs, ports, and parameters.

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

You will be presented with a menu:
1.  **Laden (Charge):** Starts the charging process.
2.  **EIS Analyse:** Runs the impedance analysis on the data.
3.  **Beenden (Exit):** Exits the program.

## Configuration

Edit `Defines.py` to configure:
*   IP Addresses (`IP_MASTER`, `IP_CLIENT1`, `IP_CLIENT2`)
*   Ports
*   Charging parameters (`CHARGE_VOLTAGE`, `CHARGE_CURRENT`)

## Requirements

*   Python 3.x
*   numpy
*   pandas
*   matplotlib
*   scipy
