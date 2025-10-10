import threading

# Global serial connection
serial_lock = threading.Lock()
ser_instance = None

def get_serial_connection():
    global ser_instance
    with serial_lock:
        if ser_instance is None or not ser_instance.is_open:
            try:
                ser_instance = serial.Serial("COM8", 115200, timeout=1)
                print("Koneksi serial COM8 dibuka")
            except Exception as e:
                print(f"Gagal buka koneksi serial: {e}")
                return None
        return ser_instance

def yolo_camera_handler(camera_index, yolo_model):
    print(f"YOLO Camera Handler started for camera {camera_index}")
    
    try:
        yolo_cam = init_camera(camera_index, camera_name="YOLO Camera")
        
        while True:
            try:
                ser = get_serial_connection()
                if ser is None:
                    time.sleep(2)
                    continue
                    
                ser.flushInput()
                
                if ser.in_waiting:
                    line = ser.readline().decode().strip()
                    print(f"Serial data: {line}")
                    
                    if "JEPRET" in line.upper():
                        print("Sinyal JEPRET diterima! Mengambil gambar dengan YOLO...")
                        
                        ok, frame = yolo_cam.read()
                        if ok and frame is not None:
                            processed_frame, detection_count = process_frame_with_yolo(frame, yolo_model)
                            
                            os.makedirs("yolo_captures", exist_ok=True)
                            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                            fname = f"yolo_detection_{timestamp}.jpg"
                            fpath = os.path.join("yolo_captures", fname)
                            
                            cv2.imwrite(fpath, processed_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                            print(f"Gambar YOLO tersimpan: {fpath}")
                            
                            yolo_caption = f"YOLO DETECTION\nDetections: {detection_count}\nTime: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                            send_telegram_photos(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, [fpath], yolo_caption)
                        else:
                            print("Gagal membaca frame dari kamera YOLO")
                
                time.sleep(0.1)
                
            except Exception as e:
                print(f"Error in YOLO handler: {e}")
                time.sleep(1)
                
    except Exception as e:
        print(f"Failed to initialize YOLO camera: {e}")

def tunggu_perintah_dari_arduino(timeout=60):
    try:
        ser = get_serial_connection()
        if ser is None:
            return False

        ser.flushInput()
        print("Menunggu sinyal 'button' dari nodemcu...")
        start = time.time()

        while time.time() - start < timeout:
            if ser.in_waiting:
                line = ser.readline().decode().strip()
                print(f"Dari nodemcu: {line}")
                if "button" in line.lower():
                    return True
            time.sleep(0.1)
        return False
    except Exception as e:
        print("Gagal konek ke NODEMCU:", e)
        return False