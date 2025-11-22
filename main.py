import Comm_test
import Ladeskript_1_0
import Impedance_Calculator1
import socket
import time
from Defines import (INITIALIZING, CHARGING, EISTEST, GETTESTDATA, IP_MASTER, IP_CLIENT1, DEFAULT_PORT)

# Protokoll of code changes:
# 19.05.2025 Malte
# Code wurde getestet und funktionierte nicht. Deswegen rückbaue auf den alten Code der funktioniert hat. Neue Sachen wurden auskommentiert.
# Bitte nichts davon löschen der Code hat den richtigen Ansatz aber probleme müssen behoben werden.
# 26.05.2025 Malte
# Code wurde getestet und funktioniert wie geplannt. Die IP-Addressen und Ports wurden auf die PowerSupply angepasst wie in der Besprechung gezeigt wurde
# Die Initialise Funktion wird jetzt für alles genutzt was vor der state maschine gemacht werden muss
# 30.05.2025 Malte
# Mit Max abgesprochen, dass Funktionen und Parameter aus Ladeskript_1_0.py in Comm_test.py und Defines.py verschoben werden.
# Desweiteren wurde die Ladefunktion und die Verbindungsfunktion in die bereits vorhandenen Funktionen integriert.
# 02.06.2025 Yves, Malte
# Code vom 30.05 getestet und funktioniert. 
# 11.06.2025 Malte, Nikita
# Code wurde getestet und Komunikation wurde erfolgreich hergestellt. Zu testen ist noch die Ladefunktion.
# 22.11.2025 Nikita
# EIS Test wurde implementiert und getestet.




# Initialisierung und Verbindungsaufbau
Comm_test.initialise()

if Comm_test.s1 is None:
    print("Client 1 connection failed, exiting...")
    time.sleep(5)
    exit(1)

if Comm_test.s2 is None:
    print("Power Supply connection failed, exiting...")
    time.sleep(5)
    exit(1)

while True:
    print("\nWas möchten Sie tun?")
    print("1: Laden")
    print("2: EIS Analyse")
    print("3: Beenden")
    wahl = input("Bitte wählen Sie eine Option (1-3): ")

    if wahl == "1":
        try:
            Ladeskript_1_0.laden(Comm_test.s2)
        except Exception as e:
            print("Charging failed:", e)
    elif wahl == "2":
        try:
            Impedance_Calculator1.main()
        except Exception as e:
            print("EIS Analysis failed:", e)
    elif wahl == "3":
        print("Programm wird beendet.")
        break
    else:
        print("Ungültige Auswahl. Bitte erneut versuchen.")