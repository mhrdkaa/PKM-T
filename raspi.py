import os, time, requests, cv2, psycopg2
from datetime import datetime
from decimal import Decimal, InvalidOperation
import serial
import json
import threading
import numpy as np
from ultralytics import YOLO
import supervision as sv
import subprocess
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

class RobustSerial:
    def __init__(self, port=None, baudrate=115200, timeout=1):
        self.port = port or IOT_PORT
        self.baudrate = baudrate
        self.timeout = timeout
        self.ser = None
        self.last_error_time = 0
        self.error_count = 0
        self.setup_serial()
    
    def setup_serial(self):
        """Setup serial connection dengan error handling"""
        try:
            # Close existing connection
            if self.ser and self.ser.is_open:
                self.ser.close()
                time.sleep(0.5)
            
            print(f"Mencoba koneksi serial ke {self.port} dengan baudrate {self.baudrate}")
            
            # Setup new connection dengan parameter optimal untuk Raspberry Pi
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
            
            # Clear buffer
            self.ser.reset_input_buffer()
            self.ser.reset_output_buffer()
            
            # Tunggu inisialisasi
            time.sleep(2)
            print(f"Serial connected to {self.port} at {self.baudrate} baud")
            self.error_count = 0
            return True
            
        except serial.SerialException as e:
            print(f"SerialException: {e}")
            self.ser = None
            return False
        except Exception as e:
            print(f"Error setup serial: {e}")
            self.ser = None
            return False
    
    def read_serial_data(self):
        """Baca data dari serial dengan robust error handling"""
        if self.ser is None or not self.ser.is_open:
            print("Serial not connected, attempting reconnect...")
            if not self.setup_serial():
                return ""
        
        try:
            # Check if data available
            if self.ser.in_waiting > 0:
                data_buffer = ""
                bytes_to_read = self.ser.in_waiting
                
                # Read available data
                byte_data = self.ser.read(bytes_to_read)
                
                try:
                    decoded_data = byte_data.decode('utf-8', errors='ignore')
                    data_buffer = decoded_data.strip()
                    
                    if data_buffer:
                        print(f"RAW SERIAL DATA: '{data_buffer}'")
                        return data_buffer
                        
                except Exception as decode_error:
                    print(f"Decode error: {decode_error}")
                    return ""
            
            return ""
            
        except serial.SerialException as e:
            print(f"SerialException during read: {e}")
            self.handle_serial_error()
            return ""
        except OSError as e:
            print(f"OSError (USB disconnect?): {e}")
            self.handle_serial_error()
            return ""
        except Exception as e:
            print(f"Unexpected error during read: {e}")
            return ""
    
    def write_serial_data(self, data):
        """Tulis data ke serial dengan error handling"""
        if self.ser is None or not self.ser.is_open:
            print("Serial not connected for writing, attempting reconnect...")
            if not self.setup_serial():
                return False
        
        try:
            # Ensure data ends with newline
            if not data.endswith('\n'):
                data += '\n'
            
            self.ser.write(data.encode())
            self.ser.flush()
            print(f"Serial write: '{data.strip()}'")
            return True
            
        except serial.SerialException as e:
            print(f"SerialException during write: {e}")
            self.handle_serial_error()
            return False
        except Exception as e:
            print(f"Error during serial write: {e}")
            return False
    
    def handle_serial_error(self):
        """Handle serial error dengan exponential backoff"""
        self.error_count += 1
        current_time = time.time()
        
        # Exponential backoff: tunggu lebih lama setelah error berulang
        backoff_time = min(2 ** self.error_count, 30)
        
        if current_time - self.last_error_time > 60:
            self.error_count = 0
        
        print(f"Serial error count: {self.error_count}, waiting {backoff_time}s before reconnect")
        time.sleep(backoff_time)
        
        self.last_error_time = current_time
        self.setup_serial()
    
    def close(self):
        """Tutup koneksi serial"""
        try:
            if self.ser and self.ser.is_open:
                self.ser.close()
                print("Serial connection closed")
        except:
            pass

# Global serial instance
serial_reader = None

def get_serial_connection():
    """Dapatkan koneksi serial shared"""
    global serial_reader
    if serial_reader is None:
        serial_reader = RobustSerial(IOT_PORT, IOT_BAUD)
    return serial_reader

def read_serial_data():
    """Wrapper untuk baca data serial"""
    global serial_reader
    if serial_reader is None:
        serial_reader = get_serial_connection()
    return serial_reader.read_serial_data()

def send_serial_data(data):
    """Wrapper untuk kirim data serial"""
    global serial_reader
    if serial_reader is None:
        serial_reader = get_serial_connection()
    return serial_reader.write_serial_data(data)

def close_serial_connection():
    """Tutup koneksi serial dengan aman"""
    global serial_reader
    if serial_reader:
        serial_reader.close()
        serial_reader = None

def fix_serial_permissions():
    """Fix permission issues di Raspberry Pi"""
    try:
        subprocess.run(['sudo', 'chmod', '666', IOT_PORT], check=True)
        print("Serial permissions fixed")
    except Exception as e:
        print(f"Permission fix error: {e}")

def diagnose_serial_issue():
    """Diagnose serial issues di Raspberry Pi"""
    print("Diagnosing serial issues...")
    
    try:
        # Cek jika device exists
        if os.path.exists(IOT_PORT):
            print(f"Device {IOT_PORT} exists")
            
            # Cek permissions
            import stat
            st = os.stat(IOT_PORT)
            permissions = stat.S_IMODE(st.st_mode)
            print(f"Permissions: {oct(permissions)}")
            
            if permissions != 0o666:
                print("Permission mungkin kurang, trying to fix...")
                fix_serial_permissions()
        else:
            print(f"Device {IOT_PORT} not found!")
            
        # Cek USB devices
        result = subprocess.run(['lsusb'], capture_output=True, text=True)
        print("USB Devices:")
        print(result.stdout if result.stdout else "No output")
        
    except Exception as e:
        print(f"Diagnosis error: {e}")

def get_serial_connection_old():
    """Dapatkan koneksi serial shared - legacy version"""
    global ser_instance
    with serial_lock:
        if ser_instance is None or not ser_instance.is_open:
            try:
                # Fix permissions first
                fix_serial_permissions()
                ser_instance = serial.Serial(IOT_PORT, IOT_BAUD, timeout=1)
                print(f"Koneksi serial {IOT_PORT} dibuka")
                time.sleep(2)
            except Exception as e:
                print(f"Gagal buka koneksi serial: {e}")
                return None
        return ser_instance

def read_serial_data_old():
    """Baca semua data yang available dari serial - legacy version"""
    ser = get_serial_connection_old()
    if ser is None:
        return ""
    try:
        data_buffer = ""
        while ser.in_waiting > 0:
            byte_data = ser.read(ser.in_waiting)
            try:
                data_buffer += byte_data.decode('utf-8', errors='ignore')
            except:
                pass
        if data_buffer:
            print(f"RAW SERIAL DATA: '{data_buffer}'")
            return data_buffer.strip()
        return ""
    except Exception as e:
        print(f"Error baca serial: {e}")
        return ""

def send_serial_data_old(data):
    """Kirim data melalui serial connection - legacy version"""
    ser = get_serial_connection_old()
    if ser is None:
        return False
    try:
        ser.write(f"{data}\n".encode())
        return True
    except Exception as e:
        print(f"Error kirim serial: {e}")
        return False

def handle_telegram_callbacks():
    """Thread untuk handle callback dari inline button Telegram"""
    print("Memulai thread callback handler...")
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
                            
                            print(f"CALLBACK DITERIMA: {callback_data}")
                            
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
                            
                            # Kirim ke Arduino via serial
                            print(f"Mengirim {kisi_command} ke Arduino...")
                            send_serial_data(kisi_command)
            
            time.sleep(1)
            
        except Exception as e:
            print(f"Error dalam handle_telegram_callbacks: {e}")
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
    """Cek apakah paket termasuk COD"""
    if not status_paket:
        return False
    
    status_normalized = str(status_paket).upper().strip()
    print(f"DEBUG is_paket_cod: input='{status_paket}', normalized='{status_normalized}', result={status_normalized == 'COD'}")
    
    return status_normalized == "COD"

def init_yolo_model(model_path="PKM-KC.pt"):
    """Initialize YOLO model"""
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
    """Process frame dengan YOLO detection - optimized untuk Raspberry Pi"""
    if model is None:
        print("YOLO model is None, returning original frame")
        return frame, 0
    
    try:
        # Resize frame untuk performa lebih baik di Raspberry Pi
        original_height, original_width = frame.shape[:2]
        if original_width > 640:
            frame = cv2.resize(frame, (640, 480))
            frame_width, frame_height = 640, 480
        
        # Initialize annotators
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

        # Run YOLO inference dengan optimasi untuk Raspberry Pi
        results = model(frame, 
                       agnostic_nms=True, 
                       verbose=False,
                       imgsz=320,
                       conf=0.5,
                       half=False
                      )[0]
        
        detections = sv.Detections.from_yolov8(results)
        
        # Create labels
        labels = [
            f"{model.model.names[class_id]} {confidence:0.2f}"
            for _, confidence, class_id, _
            in detections
        ]
        
        # Annotate frame
        annotated_frame = box_annotator.annotate(
            scene=frame.copy(), 
            detections=detections, 
            labels=labels
        )

        zone.trigger(detections=detections)
        annotated_frame = zone_annotator.annotate(scene=annotated_frame)
        
        # Add detection count info
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
    """Deteksi kamera yang tersedia"""
    available_cameras = []
    print("Mencari kamera yang tersedia...")
    
    for i in range(max_test):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                available_cameras.append(i)
                print(f"Kamera {i} ditemukan - Resolusi: {frame.shape[1]}x{frame.shape[0]}")
            else:
                print(f"Kamera {i} terdeteksi tapi tidak bisa baca frame")
            cap.release()
        else:
            print(f"Kamera {i} tidak terdeteksi")
        time.sleep(0.5)
    
    return available_cameras

def init_camera(index=0, width=640, height=480, warmup_frames=8, camera_name="Kamera"):
    """Initialize kamera dengan konfigurasi untuk Raspberry Pi"""
    cam = cv2.VideoCapture(index)
    if not cam.isOpened():
        raise RuntimeError(f"{camera_name} {index} tidak dapat dibuka.")
    
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    cam.set(cv2.CAP_PROP_FPS, 15)
    cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    # Warmup kamera
    for i in range(warmup_frames):
        ret, frame = cam.read()
        if ret:
            print(f"{camera_name} warmup {i+1}/{warmup_frames} - Brightness: {frame.mean():.1f}")
        time.sleep(0.05)
    
    # Verifikasi resolusi
    actual_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"{camera_name} {index} siap - Resolusi: {actual_width}x{actual_height}")
    
    return cam

def capture_normal_frame(cam, save_dir="captures", prefix="normal", overlay_text=None):
    """Capture frame normal tanpa YOLO"""
    os.makedirs(save_dir, exist_ok=True)
    
    ok, frame = cam.read()
    if not ok or frame is None:
        raise RuntimeError("Gagal membaca frame dari kamera.")
    
    # Add overlay text
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
    """Capture frame dengan YOLO processing"""
    os.makedirs(save_dir, exist_ok=True)
    
    ok, frame = yolo_cam.read()
    if not ok or frame is None:
        raise RuntimeError("Gagal membaca frame dari kamera YOLO.")
    
    print(f"Capturing YOLO frame - Brightness: {frame.mean():.1f}")
    
    # Process dengan YOLO
    processed_frame, detection_count = process_frame_with_yolo(frame, yolo_model)
    
    fname = f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    fpath = os.path.join(save_dir, fname)
    ok = cv2.imwrite(fpath, processed_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    
    if not ok:
        raise RuntimeError("Gagal menyimpan gambar YOLO.")
    
    print(f"YOLO frame saved - Detections: {detection_count}")
    return fpath, detection_count

def send_telegram_photos(bot_token: str, chat_id: str, image_paths: list, caption: str = "") -> bool:
    """Kirim multiple gambar ke Telegram"""
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
                print(f"Foto terkirim: {os.path.basename(img_path)}")
                success_count += 1
            else:
                print(f"Gagal kirim foto: {r.status_code}")
                
        except Exception as e:
            print(f"Error kirim foto: {e}")
    
    return success_count > 0

def send_telegram_buttons(bot_token: str, chat_id: str, resi_code: str, barang: str, harga: str):
    """Kirim inline button"""
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    
    # Buat inline keyboard
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
            print("Inline button berhasil dikirim!")
            return True
        else:
            print("Gagal kirim button:", response.status_code, response.text)
            return False
            
    except Exception as e:
        print("Error kirim button:", e)
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
    """Cari data di tabel paket berdasarkan no_resi"""
    cursor.execute("SELECT * FROM paket WHERE no_resi = %s LIMIT 1", (resi_code,))
    result = cursor.fetchone()
    if result:
        columns = [desc[0] for desc in cursor.description]
        return dict(zip(columns, result))
    return None

def test_telegram_connection():
    """Test koneksi ke Telegram"""
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getMe"
    try:
        response = requests.get(url, timeout=10, verify=False)
        if response.status_code == 200:
            print("Koneksi Telegram BOT OK")
            return True
        else:
            print(f"Telegram Bot Error: {response.status_code}")
            return False
    except Exception as e:
        print(f"Error koneksi Telegram: {e}")
        return False

def tunggu_perintah_dari_arduino(timeout=60):
    """Tunggu sinyal 'button' dari Arduino via serial"""
    try:
        print("Menunggu sinyal 'button' dari nodemcu...")
        start = time.time()
        last_status_time = time.time()

        while time.time() - start < timeout:
            data = read_serial_data()
            if data:
                print(f"Serial received: '{data}'")
                if "button" in data.lower():
                    print("Sinyal button diterima dari Arduino!")
                    return True
                elif "error" in data.lower():
                    print(f"Arduino reported error: {data}")
            
            # Print status every 10 seconds
            if time.time() - last_status_time > 10:
                elapsed = time.time() - start
                print(f"Still waiting... {elapsed:.1f}s elapsed")
                last_status_time = time.time()
                
            time.sleep(0.5)
            
        print("Timeout - Tidak ada sinyal 'button' dari Arduino")
        return False
        
    except Exception as e:
        print(f"Gagal baca dari Arduino: {e}")
        return False

if __name__ == "__main__":
    conn = cursor = None
    main_camera = None
    yolo_camera = None
    
    try:
        print("Starting application on Raspberry Pi...")
        
        # Diagnose serial first
        diagnose_serial_issue()
        
        if not test_telegram_connection():
            print("Tidak bisa melanjutkan, bot Telegram tidak bisa diakses")
            exit(1)
        
        print("Loading YOLO model...")
        yolo_model = init_yolo_model("PKM-KC.pt")
        
        if yolo_model is None:
            print("YOLO model gagal di-load, sistem tetap berjalan tanpa YOLO")
        
        # Initialize serial connection early
        print("Initializing serial connection...")
        get_serial_connection()
        
        callback_thread = threading.Thread(target=handle_telegram_callbacks, daemon=True)
        callback_thread.start()
        print("Thread callback handler started")
        
        available_cams = list_available_cameras()
        if not available_cams:
            print("Tidak ada kamera yang terdeteksi!")
            exit(1)
        
        print(f"Kamera yang tersedia: {available_cams}")
        main_camera = init_camera(available_cams[0], camera_name="Main Camera")
        
        if len(available_cams) > 1:
            try:
                yolo_camera = init_camera(available_cams[1], camera_name="YOLO Camera")
                print("Kamera YOLO siap (kamera terpisah)")
            except Exception as e:
                print(f"Gagal inisialisasi kamera YOLO terpisah: {e}")
                yolo_camera = main_camera
                print("Menggunakan kamera utama untuk YOLO")
        else:
            print("Hanya 1 kamera terdeteksi - menggunakan kamera utama untuk YOLO juga")
            yolo_camera = main_camera
            
        conn, cursor = get_db_cursor()
        print("Koneksi PostgreSQL OK.")
        print("Sistem siap!")
        print("Silakan scan QR Code... (CTRL+C untuk berhenti)")
        
        while True:
            data = input().strip()
            if not data: 
                continue
            print("Hasil Scan:", data)
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
                    
                    print(f"Gambar utama tersimpan: {img_path}")
                    serial_success = send_serial_data("buka_1")
                    photo_success = send_telegram_photos(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, [img_path], caption=caption)
                    
                    if photo_success:
                        print("Foto utama berhasil terkirim ke Telegram.")
                        
                        if is_cod:
                            print("Paket COD - Menunggu sinyal button dari Arduino...")
                            
                            if tunggu_perintah_dari_arduino():
                                print("Mengambil gambar YOLO...")
                                
                                if yolo_model is not None:
                                    yolo_path, detection_count = capture_yolo_frame(yolo_camera, yolo_model)
                                    print(f"Gambar YOLO tersimpan: {yolo_path}")
                                    
                                    yolo_caption = f"YOLO DETECTION\nDetections: {detection_count}\nTime: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                                    yolo_success = send_telegram_photos(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, [yolo_path], yolo_caption)
                                    
                                    if yolo_success:
                                        print("Gambar YOLO berhasil dikirim")
                                        print("Mengirim inline button ke Telegram...")
                                        button_success = send_telegram_buttons(
                                            TELEGRAM_BOT_TOKEN, 
                                            TELEGRAM_CHAT_ID, 
                                            resi_val or data, 
                                            barang_val or "-",
                                            harga_display
                                        )
                                        
                                        if button_success:
                                            print("Inline button berhasil dikirim!")
                                            print("User bisa klik button KISI 1 atau KISI 2")
                                        else:
                                            print("Gagal kirim button")
                                    else:
                                        print("Gagal kirim gambar YOLO")
                                else:
                                    print("YOLO model tidak tersedia, langsung kirim button")
                                    button_success = send_telegram_buttons(
                                        TELEGRAM_BOT_TOKEN, 
                                        TELEGRAM_CHAT_ID, 
                                        resi_val or data, 
                                        barang_val or "-",
                                        harga_display
                                    )
                            else:
                                print("Tidak ada sinyal button dari Arduino")
                        else:
                            print("Paket NON COD - Proses selesai")
                    
                    else:
                        print("Gagal kirim foto utama")
                    
                except Exception as e:
                    print("Gagal proses pengiriman:", e)
            else:
                print("Resi tidak ditemukan di database")

    except KeyboardInterrupt:
        print("\nProgram dihentikan oleh pengguna.")
    except Exception as e:
        print("Error utama:", e)
    finally:
        try:
            if cursor: cursor.close()
            if conn: conn.close()
        except: pass
        
        if main_camera and main_camera.isOpened():
            main_camera.release()
            print("Kamera utama di-release")
        
        if yolo_camera and yolo_camera.isOpened() and yolo_camera != main_camera:
            yolo_camera.release()
            print("Kamera YOLO di-release")
        
        close_serial_connection()