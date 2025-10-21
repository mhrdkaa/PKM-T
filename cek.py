import serial, time

ser = serial.Serial("/dev/arduino", 115200, timeout=1)
time.sleep(2)  # tunggu Arduino reboot

print("Koneksi OK. Membaca data...")
while True:
    if ser.in_waiting > 0:
        print(ser.readline().decode(errors='ignore').strip())
