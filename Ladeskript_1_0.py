import socket
import time
from Defines import (IP_MASTER, IP_CLIENT1, IP_CLIENT2, DEFAULT_PORT, CLIENT1_PORT, POWER_SUPPLY_PORT, CONNECTION_TIMEOUT, RESPONSE_TIMEOUT, MAX_RETRIES)
from Defines import (BATTERY_VOLTAGE_LIMIT, BATTERY_CURRENT_MIN, CHARGE_CURRENT, CHARGE_VOLTAGE, CHECK_INTERVAL, TIMEOUT_MINUTES)
import Comm_test

# Funktion für die Aktuelle Sapnnung und Strom fals noch nicht vorhandne auf RP1
def get_battery_voltage():
    return 13.8

def get_battery_current():
    return 2.5

# --- SCPI senden ---
def scpi_send(sock, cmd):
    sock.sendall((cmd + '\n').encode('ascii'))

# --- Ladefunktion ---
def laden(sock):
    try:
        # Konfiguration der Ladeparameter
        scpi_send(sock, "*RST")
        scpi_send(sock, "SYST:REM")
        scpi_send(sock, "OUTP:STAT OFF")
        scpi_send(sock, "BATT:MODE CHARG")
        scpi_send(sock, f"VOLT {CHARGE_VOLTAGE}")
        scpi_send(sock, f"CURR {CHARGE_CURRENT}")
        scpi_send(sock, f"BATT:STOP:CURR {BATTERY_CURRENT_MIN}")
        scpi_send(sock, f"BATT:STOP:TIME {TIMEOUT_MINUTES}")
        

        # Überwachung der Ladekonfiguration 
        start = time.time()
        i = 0
        while ( i==0):
            scpi_send(sock, "OUTP:STAT ON")
            voltage = get_battery_voltage()
            current = get_battery_current()
            print(f"[{time.strftime('%H:%M:%S')}] Spannung: {voltage:.2f} V | Strom: {current:.2f} A")

            if voltage >= BATTERY_VOLTAGE_LIMIT:
                print("Ladestopp: Spannung erreicht.")
                break
            if current <= BATTERY_CURRENT_MIN:
                print("Ladestopp: Strom unter Mindestwert.")
                break
            if (time.time() - start) > TIMEOUT_MINUTES * 60:
                print("Ladestopp: Maximale Ladezeit erreicht.")
                scpi_send(sock, "OUTP:STAT OFF")
                i = 1
                break

            time.sleep(CHECK_INTERVAL) 

        #scpi_send(sock, "OUTP:STAT OFF")
        print("Ladevorgang abgeschlossen.") 
    except Exception as e:
        print(f"Fehler beim Laden: {e}")

