import socket
from Defines import (IP_MASTER, IP_CLIENT1, IP_CLIENT2, DEFAULT_PORT,CLIENT1_PORT,POWER_SUPPLY_PORT, CONNECTION_TIMEOUT, RESPONSE_TIMEOUT, MAX_RETRIES)

# global soccets for each client
s1 = None
s2 = None
# Connection status flags for each client
isconnected_Client1 = False # Raspberry Pi connected to BMS
isconnected_Client2 = False # Power Supply has to be implemented

# Simple function to connect to a client via IP and Port 
def open_connection(ip, port):
    try:
        # Create a socket object
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind((ip, port))
        s.listen(1)
        s.settimeout(CONNECTION_TIMEOUT)
        conn, addr = s.accept()
        print("Connectd by:", addr)
        data = conn.recv(1024)
        return True, conn
    except socket.timeout:
        print("Connection timed out")
        s.close()
        return False, None
    except Exception as e:
        print("Error:", e)
        s.close()
        return False, None
    
def scpi_send(sock, cmd):
    sock.sendall((cmd + '\n').encode('ascii'))    

# --- Verbindung herstellen + testen mit Power Supply ---
def connect_to_power_supply(ip, port):
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((ip, port))
        sock.settimeout(CONNECTION_TIMEOUT)
        print("Verbindung aufgebaut.")

        # Verbindung testen mit *IDN?
        sock.sendall(b'*IDN?\n')
        antwort = sock.recv(1024).decode('ascii').strip()
        print(f"Gerät antwortet: {antwort}")
        return True, sock
    except socket.timeout:
        print("Verbindung zur Power Supply timed out")
        sock.close()
        return False, None
    except Exception as e:
        print(f"Verbindungsfehler: {e}")
        sock.close()
        return False, None   

# Only features the connection to Client 1. Client 2 has to be implemented. Maybe we dont need this funktion at all.
def open_connection_to_all_clients():
    global isconnected_Client1, isconnected_Client2
    global s1, s2
    # retries for each client
    retriesClient1 = 0
    retriesClient2 = 0
    success=False
    # Check connection to Client 1
    while retriesClient1 < MAX_RETRIES:
        success, s1=open_connection(IP_MASTER, CLIENT1_PORT)
        if success:
            print("Connection to Client1 successful")
            isconnected_Client1 = True
            break
        else:
            print("Connection to CLient1 failed, retrying...")
            retriesClient1 += 1
    if retriesClient1 == MAX_RETRIES:
        print("Max retries reached, unable to connect to Client 1")
        isconnected_Client1 = False      
    # Check connection to Client 2
    success = False
    while retriesClient2 < MAX_RETRIES:
        print("trying to connect to Client 2")
        success, s2= connect_to_power_supply(IP_CLIENT2, POWER_SUPPLY_PORT)
        print("Connection to Client2:", success)
        if success:
            print("Connection to Client2 successful")
            isconnected_Client2 = True
            break
        else:
            print("Connection to Client2 failed, retrying...")
            retriesClient2 += 1
    
    if retriesClient2 >= MAX_RETRIES:
        print("Max retries reached, unable to connect to Client 2")
        isconnected_Client2 = False   


# Initialise the connections to the clients
def initialise():
    global isconnected_Client1, isconnected_Client2
    isconnected_Client1 = False
    isconnected_Client2 = False
    open_connection_to_all_clients()
    if isconnected_Client1 and isconnected_Client2:
        print("All clients connected successfully")


# Send command to the client and receive a response
# These are the commands that can be sent to the client:
# command = 0 => PowerSupply current charging parameters
# command = 1 => PowerSupply future charging parameters
# command = 2 => EIS Test Data
def get_data(s, command, number_bytes_to_send, buffer_response, number_bytes_to_receive, timeout=RESPONSE_TIMEOUT):
    try:
        # Send data
        s.sendall(command)
    
        # Receive response
        s.settimeout(timeout)
        response = s.recv(number_bytes_to_receive)
        
        # Process response
        buffer_response = response[:number_bytes_to_receive]
        return buffer_response
    except socket.timeout:
        print("Response timed out")
        return None
    except Exception as e:
        print("Error:", e)
        return None
