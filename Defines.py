# Define the IP addresses (Müssen lokale IP-Adressen sein)
IP_MASTER = "Place the IP Address of Master Raspberry pi here"
IP_CLIENT1 = "Place the IP Address of Client Raspberry pi here"
IP_CLIENT2 = "Place the IP Address of Power Supply here" # (Power_SUPPLY)

# Define the ports
DEFAULT_PORT = enter port of master
CLIENT1_PORT = enter port of client
POWER_SUPPLY_PORT = enter port of power supply

# Define the timeouts and retries in ms
CONNECTION_TIMEOUT = 15
RESPONSE_TIMEOUT = 5
MAX_RETRIES = 3

# Define the states
INITIALIZING = 0
CHARGING = 1
EISTEST = 2
GETTESTDATA = 3

# Define the commands
GETCURRENTCHARGINGPARAMETERS = 0
GETFUTURECHARGINGPARAMETERS = 1
GETEISTESTDATA = 2

#Ladestation Parameter
BATTERY_VOLTAGE_LIMIT = 14.6
BATTERY_CURRENT_MIN = 0.5
CHARGE_CURRENT = 1
CHARGE_VOLTAGE = 14.6
CHECK_INTERVAL = 5
TIMEOUT_MINUTES = 1  

