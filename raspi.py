import os
import time
import requests
import psycopg2
from datetime import datetime
from decimal import Decimal, InvalidOperation
import serial
import json
import threading
import numpy as np
from ultralytics import YOLO
import supervision as sv
import subprocess
import cv2


try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

IOT_PORT = "/dev/ttyUSB0"  
IOT_BAUD = 115200
POSTGRES_HOST = os.getenv("POSTGRES_HOST")
POSTGRES_USER = os.getenv("POSTGRES_USER")
POSTGRES_PASS = os.getenv("POSTGRES_PASS")
POSTGRES_DB   = os.getenv("POSTGRES_DB")
POSTGRES_PORT = os.getenv("POSTGRES_PORT")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID")

callback_lock = threading.Lock()
serial_lock = threading.Lock()
ser_instance = None
yolo_model = None

ZONE_POLYGON = np.array([
    [0, 0],
    [1, 0],
    [1, 1],
    [0, 1]
])

class FixedRobustSerial:
    def __init__(self, port=None, baudrate=115200, timeout=1):
        self.port = port or IOT_PORT
        self.baudrate = baudrate
        self.timeout = timeout
        self.ser = None
        self.setup_serial()
    
    def setup_serial(self):
        max_retries = 3
        for attempt in range(max_retries):
            try:
                if self.ser and self.ser.is_open:
                    self.ser.close()
                    time.sleep(1)
                
                os.system(f'sudo chmod 666 {self.port}')
                time.sleep(0.5)
                
                print(f"Attempt {attempt+1}: Connecting to {self.port} at {self.baudrate}")
                
                self.ser = serial.Serial(
                    port=self.port,
                    baudrate=self.baudrate,
                    timeout=self.timeout,
                    write_timeout=self.timeout,
                    bytesize=serial.EIGHTBITS,
                    parity=serial.PARITY_NONE,
                    stopbits=serial.STOPBITS_ONE,
                    rtscts=False,
                    dsrdtr=False,
                    xonxoff=False
                )
                
                self.ser.dtr = False
                time.sleep(0.5)
                self.ser.dtr = True
                time.sleep(2)
                
                self.ser.reset_input_buffer()
                self.ser.reset_output_buffer()
                
                print(f"Serial connected: {self.port}")
                return True
                
            except serial.SerialException as e:
                print(f"Serial attempt {attempt+1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                return False
            except Exception as e:
                print(f"Unexpected error: {e}")
                return False
    
    def read_serial_data(self):
        if self.ser is None or not self.ser.is_open:
            print("Serial not connected, reconnecting...")
            if not self.setup_serial():
                return ""
        
        try:
            if self.ser.in_waiting > 0:
                data = self.ser.readline().decode('utf-8', errors='ignore').strip()
                if data:
                    print(f"Serial received: '{data}'")
                    return data
            return ""
            
        except serial.SerialException as e:
            print(f"Serial read error: {e}")
            self.ser = None
            return ""
        except Exception as e:
            print(f"Read error: {e}")
            return ""
    
    def write_serial_data(self, data):
        if self.ser is None or not self.ser.is_open:
            if not self.setup_serial():
                return False
        
        try:
            if not data.endswith('\n'):
                data += '\n'
                
            self.ser.write(data.encode())
            self.ser.flush()
            print(f"Serial sent: '{data.strip()}'")
            return True
            
        except serial.SerialException as e:
            print(f"Serial write error: {e}")
            self.ser = None
            return False
        except Exception as e:
            print(f"Write error: {e}")
            return False
    
    def close(self):
        try:
            if self.ser and self.ser.is_open:
                self.ser.close()
                print("Serial closed")
        except:
            pass

serial_reader = None

def get_serial_connection():
    global serial_reader
    if serial_reader is None:
        serial_reader = FixedRobustSerial(IOT_PORT, IOT_BAUD)
    return serial_reader

def read_serial_data():
    global serial_reader
    if serial_reader is None:
        serial_reader = get_serial_connection()
    return serial_reader.read_serial_data()

def send_serial_data(data):
    global serial_reader
    if serial_reader is None:
        serial_reader = get_serial_connection()
    return serial_reader.write_serial_data(data)

def close_serial_connection():
    global serial_reader
    if serial_reader:
        serial_reader.close()
        serial_reader = None

def fix_serial_permissions():
    try:
        subprocess.run(['sudo', 'chmod', '666', IOT_PORT], check=True)
        print("Serial permissions fixed")
    except Exception as e:
        print(f"Permission fix error: {e}")

def diagnose_serial_issue():
    print("Diagnosing serial issues...")
    
    try:
        if os.path.exists(IOT_PORT):
            print(f"Device {IOT_PORT} exists")
            
            import stat
            st = os.stat(IOT_PORT)
            permissions = stat.S_IMODE(st.st_mode)
            print(f"Permissions: {oct(permissions)}")
            
            if permissions != 0o666:
                print("Fixing permissions...")
                fix_serial_permissions()
        else:
            print(f"Device {IOT_PORT} not found!")
            
        result = subprocess.run(['lsusb'], capture_output=True, text=True)
        print("USB Devices:")
        print(result.stdout if result.stdout else "No output")
        
    except Exception as e:
        print(f"Diagnosis error: {e}")

def handle_telegram_callbacks():
    print("Starting callback handler thread...")
    last_update_id = 0
    
    while True:
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getUpdates"
            params = {"timeout": 30, "offset": last_update_id + 1}
            
            response = requests.get(url, params=params, timeout=35, verify=False)
            
            if response.status_code == 200:
                data = response.json()
                if data.get("ok") and data.get("result"):
                    for update in data["result"]:
                        last_update_id = update["update_id"]
                        
                        if "callback_query" in update:
                            callback_data = update["callback_query"]["data"]
                            message_id = update["callback_query"]["message"]["message_id"]
                            chat_id = update["callback_query"]["message"]["chat"]["id"]
                            
                            print(f"CALLBACK RECEIVED: {callback_data}")
                            
                            answer_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/answerCallbackQuery"
                            answer_data = {
                                "callback_query_id": update["callback_query"]["id"],
                                "text": f"Pilihan diterima: {callback_data}",
                                "show_alert": False
                            }
                            requests.post(answer_url, json=answer_data, verify=False)
                            
                            edit_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/editMessageText"
                            
                            if callback_data.startswith("buka_1"):
                                new_text = "Kisi 1 DIPILIH\nMengirim perintah ke sistem..."
                                kisi_command = "R1A"
                            elif callback_data.startswith("buka_2"):
                                new_text = "KISI 2 DIPILIH\nMengirim perintah ke sistem..."
                                kisi_command = "R2A"
                            elif callback_data.startswith("R3A"):
                                new_text = "Pintu Utama DIPILIH\nMengirim perintah ke sistem..."
                                kisi_command = "R3A"
                            elif callback_data.startswith("R4A"):
                                new_text = "Pintu Belakang\nMengirim perintah ke sistem..."
                                kisi_command = "R4A"
                            elif callback_data.startswith("R5A"):
                                new_text = "Pintu Atas\nMengirim perintah ke sistem..."
                                kisi_command = "R5A"
                            else:
                                continue
                            
                            edit_data = {
                                "chat_id": chat_id,
                                "message_id": message_id,
                                "text": new_text,
                                "parse_mode": "Markdown"
                            }
                            requests.post(edit_url, json=edit_data, verify=False)
                            
                            print(f"Sending {kisi_command} to Arduino...")
                            send_serial_data(kisi_command)
            
            time.sleep(1)
            
        except Exception as e:
            print(f"Error in handle_telegram_callbacks: {e}")
            time.sleep(5)

def format_rupiah(value):
    if value in (None, ""): 
        return "-"
    try:
        if not isinstance(value, Decimal):
            value = Decimal(str(value))
        s = f"{value:,.0f}".replace(",", ".")
        return s
    except (InvalidOperation, Exception):
        return str(value)

def pick_first(d: dict, keys):
    for k in keys:
        if k in d and d[k] not in (None, "", b""):
            return d[k]
    return None

def is_paket_cod(status_paket):
    if not status_paket:
        return False
    
    status_normalized = str(status_paket).upper().strip()
    print(f"DEBUG is_paket_cod: input='{status_paket}', normalized='{status_normalized}', result={status_normalized == 'COD'}")
    
    return status_normalized == "COD"

def init_yolo_model(model_path="PKM-KC.pt"):
    global yolo_model
    try:
        yolo_model = YOLO(model_path)
        print(f"YOLO model loaded: {model_path}")
        return yolo_model
    except Exception as e:
        print(f"Failed to load YOLO model: {e}")
        yolo_model = None
        return None

def process_frame_with_yolo(frame, model, frame_width=640, frame_height=480):
    if model is None:
        print("YOLO model is None, returning original frame")
        return frame, 0
    
    try:
        original_height, original_width = frame.shape[:2]
        if original_width > 640:
            frame = cv2.resize(frame, (640, 480))
            frame_width, frame_height = 640, 480
        
        box_annotator = sv.BoxAnnotator(
            thickness=1,
            text_scale=0.5
        )

        zone_polygon = (ZONE_POLYGON * np.array([frame_width, frame_height])).astype(int)
        zone = sv.PolygonZone(polygon=zone_polygon, frame_resolution_wh=(frame_width, frame_height))
        zone_annotator = sv.PolygonZoneAnnotator(
            zone=zone, 
            color=sv.Color.red(),
            thickness=1,
            text_scale=0.8
        )

        results = model(frame, 
                       agnostic_nms=True, 
                       verbose=False,
                       imgsz=320,
                       conf=0.5,
                       half=False
                      )[0]
        
        detections = sv.Detections.from_yolov8(results)
        
        labels = [
            f"{model.model.names[class_id]} {confidence:0.2f}"
            for _, confidence, class_id, _
            in detections
        ]
        
        annotated_frame = box_annotator.annotate(
            scene=frame.copy(), 
            detections=detections, 
            labels=labels
        )

        zone.trigger(detections=detections)
        annotated_frame = zone_annotator.annotate(scene=annotated_frame)
        
        detection_count = len(detections)
        cv2.putText(annotated_frame, f"Detections: {detection_count}", 
                   (10, frame_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        print(f"YOLO processing successful - Detections: {detection_count}")
        return annotated_frame, detection_count
        
    except Exception as e:
        print(f"YOLO processing error: {e}")
        cv2.putText(frame, "YOLO Error - Original Frame", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        return frame, 0

def list_available_cameras(max_test=3):
    available_cameras = []
    print("Searching for available cameras...")
    
    for i in range(max_test):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                available_cameras.append(i)
                print(f"Camera {i} found - Resolution: {frame.shape[1]}x{frame.shape[0]}")
            else:
                print(f"Camera {i} detected but cannot read frame")
            cap.release()
        else:
            print(f"Camera {i} not detected")
        time.sleep(0.5)
    
    return available_cameras

def init_camera(index=0, width=640, height=480, warmup_frames=8, camera_name="Kamera"):
    cam = cv2.VideoCapture(index)
    if not cam.isOpened():
        raise RuntimeError(f"{camera_name} {index} tidak dapat dibuka.")
    
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    cam.set(cv2.CAP_PROP_FPS, 15)
    cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    for i in range(warmup_frames):
        ret, frame = cam.read()
        if ret:
            print(f"{camera_name} warmup {i+1}/{warmup_frames} - Brightness: {frame.mean():.1f}")
        time.sleep(0.05)
    
    actual_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"{camera_name} {index} ready - Resolution: {actual_width}x{actual_height}")
    
    return cam

def capture_normal_frame(cam, save_dir="captures", prefix="normal", overlay_text=None):
    os.makedirs(save_dir, exist_ok=True)
    
    ok, frame = cam.read()
    if not ok or frame is None:
        raise RuntimeError("Gagal membaca frame dari kamera.")
    
    if overlay_text:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        text = f"{overlay_text} | {ts}"
        cv2.putText(frame, text, (18, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    
    fname = f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    fpath = os.path.join(save_dir, fname)
    ok = cv2.imwrite(fpath, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    
    if not ok:
        raise RuntimeError("Gagal menyimpan gambar.")
    
    return fpath

def capture_yolo_frame(yolo_cam, yolo_model, save_dir="yolo_captures", prefix="yolo"):
    os.makedirs(save_dir, exist_ok=True)
    
    ok, frame = yolo_cam.read()
    if not ok or frame is None:
        raise RuntimeError("Gagal membaca frame dari kamera YOLO.")
    
    print(f"Capturing YOLO frame - Brightness: {frame.mean():.1f}")
    
    processed_frame, detection_count = process_frame_with_yolo(frame, yolo_model)
    
    fname = f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    fpath = os.path.join(save_dir, fname)
    ok = cv2.imwrite(fpath, processed_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    
    if not ok:
        raise RuntimeError("Gagal menyimpan gambar YOLO.")
    
    print(f"YOLO frame saved - Detections: {detection_count}")
    return fpath, detection_count

def send_telegram_photos(bot_token: str, chat_id: str, image_paths: list, caption: str = "") -> bool:
    if not image_paths:
        return False
    
    success_count = 0
    for img_path in image_paths:
        if not os.path.exists(img_path):
            continue
            
        url = f"https://api.telegram.org/bot{bot_token}/sendPhoto"
        try:
            with open(img_path, "rb") as f:
                files = {"photo": f}
                data = {"chat_id": chat_id, "caption": caption}
                
                r = requests.post(url, data=data, files=files, timeout=30, verify=False)
            
            if r.status_code == 200:
                print(f"Photo sent: {os.path.basename(img_path)}")
                success_count += 1
            else:
                print(f"Failed to send photo: {r.status_code}")
                
        except Exception as e:
            print(f"Error sending photo: {e}")
    
    return success_count > 0

def send_telegram_buttons(bot_token: str, chat_id: str, resi_code: str, barang: str, harga: str):
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    
    keyboard = {
        "inline_keyboard": [
            [
                {"text": "KISI 1", "callback_data": f"kisi_1_{resi_code}"},
                {"text": "KISI 2", "callback_data": f"kisi_2_{resi_code}"}
            ]
        ]
    }
    
    message_text = f"**PILIH KISI UNTUK:**\nResi: `{resi_code}`\nBarang: {barang}\nHarga: Rp {format_rupiah(harga)}\n\nKlik salah satu button:"
    
    try:
        data = {
            "chat_id": chat_id,
            "text": message_text,
            "parse_mode": "Markdown",
            "reply_markup": json.dumps(keyboard)
        }
        
        response = requests.post(url, json=data, timeout=10, verify=False)
        
        if response.status_code == 200:
            print("Inline button sent successfully!")
            return True
        else:
            print("Failed to send button:", response.status_code, response.text)
            return False
            
    except Exception as e:
        print("Error sending button:", e)
        return False

def get_db_cursor():
    conn = psycopg2.connect(
        host=POSTGRES_HOST,
        user=POSTGRES_USER,
        password=POSTGRES_PASS,
        database=POSTGRES_DB,
        port=POSTGRES_PORT
    )
    return conn, conn.cursor()

def get_resi_detail(cursor, resi_code):
    cursor.execute("SELECT * FROM paket WHERE no_resi = %s LIMIT 1", (resi_code,))
    result = cursor.fetchone()
    if result:
        columns = [desc[0] for desc in cursor.description]
        return dict(zip(columns, result))
    return None

def test_telegram_connection():
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getMe"
    try:
        response = requests.get(url, timeout=10, verify=False)
        if response.status_code == 200:
            print("Telegram BOT connection OK")
            return True
        else:
            print(f"Telegram Bot Error: {response.status_code}")
            return False
    except Exception as e:
        print(f"Telegram connection error: {e}")
        return False

def tunggu_perintah_dari_arduino(timeout=60):
    try:
        print("Waiting for 'button' signal from nodemcu...")
        start = time.time()
        last_status_time = time.time()

        while time.time() - start < timeout:
            data = read_serial_data()
            if data:
                print(f"Serial received: '{data}'")
                if "button" in data.lower():
                    print("Button signal received from Arduino!")
                    return True
                elif "error" in data.lower():
                    print(f"Arduino reported error: {data}")
            if time.time() - last_status_time > 10:
                elapsed = time.time() - start
                print(f"Still waiting... {elapsed:.1f}s elapsed")
                last_status_time = time.time()
                
            time.sleep(0.5)
            
        print("Timeout - No 'button' signal from Arduino")
        return False
        
    except Exception as e:
        print(f"Failed to read from Arduino: {e}")
        return False

if __name__ == "__main__":
    conn = cursor = None
    main_camera = None
    yolo_camera = None
    
    try:
        print("Starting application on Raspberry Pi...")
        diagnose_serial_issue()
        
        if not test_telegram_connection():
            print("Cannot continue, Telegram bot not accessible")
            exit(1)
        
        print("Loading YOLO model...")
        yolo_model = init_yolo_model("PKM-KC.pt")
        
        if yolo_model is None:
            print("YOLO model failed to load, system will run without YOLO")
        
        print("Initializing serial connection...")
        time.sleep(2)
        os.system(f'sudo chmod 666 {IOT_PORT}')
        time.sleep(1)
        get_serial_connection()
        time.sleep(3)
        
        callback_thread = threading.Thread(target=handle_telegram_callbacks, daemon=True)
        callback_thread.start()
        print("Callback handler thread started")
        
        available_cams = list_available_cameras()
        if not available_cams:
            print("No cameras detected!")
            exit(1)
        
        print(f"Available cameras: {available_cams}")
        main_camera = init_camera(available_cams[0], camera_name="Main Camera")
        
        if len(available_cams) > 1:
            try:
                yolo_camera = init_camera(available_cams[1], camera_name="YOLO Camera")
                print("YOLO camera ready (separate camera)")
            except Exception as e:
                print(f"Failed to initialize separate YOLO camera: {e}")
                yolo_camera = main_camera
                print("Using main camera for YOLO")
        else:
            print("Only 1 camera detected - using main camera for YOLO too")
            yolo_camera = main_camera
            
        conn, cursor = get_db_cursor()
        print("PostgreSQL connection OK.")
        print("System ready!")
        print("Please scan QR Code... (CTRL+C to stop)")
        
        while True:
            data = input().strip()
            if not data: 
                continue
            print("Scan Result:", data)
            row = get_resi_detail(cursor, data)
            found = row is not None
            if found:
                
                resi_val   = pick_first(row, ["no_resi", "resi"])
                barang_val = pick_first(row, ["nama_paket", "barang", "nama_barang"])
                harga_raw  = pick_first(row, ["harga", "harga_barang"])
                status_val = pick_first(row, ["status_paket", "status"])
                harga_display = harga_raw if harga_raw is not None else "0"
                is_cod = is_paket_cod(status_val)
                
                cap_lines = [
                    "Resi terverifikasi",
                    f"Resi  : {resi_val or '-'}",
                    f"Paket : {barang_val or '-'}",
                    f"Harga : Rp {format_rupiah(harga_display)}",
                    f"Status: {status_val or '-'}",
                    f"Tipe  : {'COD' if is_cod else 'NON COD'}"
                ]
                caption = "\n".join(cap_lines)

                try:
                    img_path = capture_normal_frame(
                        main_camera,
                        save_dir="captures",
                        prefix="resi",
                        overlay_text=f"RESI: {resi_val or data}"
                    )
                    
                    print(f"Main image saved: {img_path}")
                    photo_success = send_telegram_photos(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, [img_path], caption=caption)
                    serial_success = send_serial_data("R5A")
                    
                    if photo_success:
                        print("Main photo successfully sent to Telegram.")
                        if is_cod:
                            print("COD package - Waiting for button signal from Arduino...")
                            
                            if tunggu_perintah_dari_arduino():
                                print("Capturing YOLO image...")
                                
                                if yolo_model is not None:
                                    yolo_path, detection_count = capture_yolo_frame(yolo_camera, yolo_model)
                                    print(f"YOLO image saved: {yolo_path}")
                                    
                                    yolo_caption = f"YOLO DETECTION\nDetections: {detection_count}\nTime: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                                    yolo_success = send_telegram_photos(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, [yolo_path], yolo_caption)
                                    
                                    if yolo_success:
                                        print("YOLO image successfully sent")
                                        print("Sending inline button to Telegram...")
                                        button_success = send_telegram_buttons(
                                            TELEGRAM_BOT_TOKEN, 
                                            TELEGRAM_CHAT_ID, 
                                            resi_val or data, 
                                            barang_val or "-",
                                            harga_display
                                        )
                                        
                                        if button_success:
                                            print("Inline button sent successfully!")
                                            print("User can click KISI 1 or KISI 2 button")
                                        else:
                                            print("Failed to send button")
                                    else:
                                        print("Failed to send YOLO image")
                                else:
                                    print("YOLO model not available, sending button directly")
                                    button_success = send_telegram_buttons(
                                        TELEGRAM_BOT_TOKEN, 
                                        TELEGRAM_CHAT_ID, 
                                        resi_val or data, 
                                        barang_val or "-",
                                        harga_display
                                    )
                            else:
                                print("No button signal from Arduino")
                        else:
                            print("NON COD package - Process completed")
                    
                    else:
                        print("Failed to send main photo")
                    
                except Exception as e:
                    print("Failed to process delivery:", e)
            else:
                print("Resi not found in database")

    except KeyboardInterrupt:
        print("\nProgram stopped by user.")
    except Exception as e:
        print("Main error:", e)
    finally:
        try:
            if cursor: cursor.close()
            if conn: conn.close()
        except: pass
        
        if main_camera and main_camera.isOpened():
            main_camera.release()
            print("Main camera released")
        
        if yolo_camera and yolo_camera.isOpened() and yolo_camera != main_camera:
            yolo_camera.release()
            print("YOLO camera released")
        
        close_serial_connection()