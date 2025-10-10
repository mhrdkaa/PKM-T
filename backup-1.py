import os, time, requests, cv2, psycopg2
from datetime import datetime
from decimal import Decimal, InvalidOperation
import serial
import json
import threading
import numpy as np
from ultralytics import YOLO
import supervision as sv
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

IOT_PORT = "COM8"
IOT_BAUD = 115200
POSTGRES_HOST = os.getenv("POSTGRES_HOST")
POSTGRES_USER = os.getenv("POSTGRES_USER")
POSTGRES_PASS = os.getenv("POSTGRES_PASS")
POSTGRES_DB   = os.getenv("POSTGRES_DB")
POSTGRES_PORT = os.getenv("POSTGRES_PORT")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "7607441123:AAE9YQU2-gFXAzWB4nnXNGpOYJyvKE41ZNY")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "5558164298")

# Global variable untuk menyimpan pesan terbaru dari Telegram
latest_telegram_message = None
telegram_lock = threading.Lock()

# Global serial connection dengan lock
serial_lock = threading.Lock()
ser_instance = None

# YOLO Configuration
ZONE_POLYGON = np.array([
    [0, 0],
    [1, 0],
    [1, 1],
    [0, 1]
])

# ==== Serial Connection Manager ====
def get_serial_connection():
    """Dapatkan koneksi serial shared"""
    global ser_instance
    with serial_lock:
        if ser_instance is None or not ser_instance.is_open:
            try:
                ser_instance = serial.Serial(IOT_PORT, IOT_BAUD, timeout=1)
                print(f"Koneksi serial {IOT_PORT} dibuka")
                time.sleep(2)
            except Exception as e:
                print(f"Gagal buka koneksi serial: {e}")
                return None
        return ser_instance

def close_serial_connection():
    """Tutup koneksi serial dengan aman"""
    global ser_instance
    with serial_lock:
        if ser_instance and ser_instance.is_open:
            ser_instance.close()
            ser_instance = None
            print("Koneksi serial ditutup")

def read_serial_data():
    """Baca semua data yang available dari serial"""
    ser = get_serial_connection()
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
            print(f"📨 RAW SERIAL DATA: '{data_buffer}'")
            return data_buffer.strip()
        return ""
    except Exception as e:
        print(f"Error baca serial: {e}")
        return ""

def send_serial_data(data):
    """Kirim data melalui serial connection"""
    ser = get_serial_connection()
    if ser is None:
        return False
    
    try:
        ser.write(f"{data}\n".encode())
        print(f"Data dikirim via serial: {data}")
        return True
    except Exception as e:
        print(f"Error kirim serial: {e}")
        return False

# ==== Telegram Message Handler ====
def handle_telegram_messages():
    """Thread untuk membaca pesan dari Telegram (termasuk /kisi_1 dari bot)"""
    global latest_telegram_message
    
    print("🔔 Memulai thread pemantau pesan Telegram...")
    last_update_id = 0
    
    while True:
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getUpdates"
            params = {"timeout": 30, "offset": last_update_id + 1}
            
            response = requests.get(url, params=params, timeout=35)
            
            if response.status_code == 200:
                data = response.json()
                if data.get("ok") and data.get("result"):
                    for update in data["result"]:
                        last_update_id = update["update_id"]
                        
                        # Cek jika ada pesan text
                        if "message" in update and "text" in update["message"]:
                            message_text = update["message"]["text"]
                            chat_id = update["message"]["chat"]["id"]
                            
                            print(f"PESAN DITERIMA: '{message_text}' dari chat {chat_id}")
                            
                            # Hanya proses pesan yang berasal dari chat_id yang diinginkan
                            if str(chat_id) == TELEGRAM_CHAT_ID:
                                if message_text.startswith('/kisi_1') or message_text.startswith('/kisi_2'):
                                    print(f"PERINTAH KISI DITERIMA: {message_text}")
                                    
                                    # Simpan pesan untuk diproses
                                    with telegram_lock:
                                        latest_telegram_message = message_text
                                    
                                    # Langsung kirim ke Arduino
                                    kisi_command = message_text.replace('/', '')  # Hapus slash
                                    send_serial_data(kisi_command)
            
            time.sleep(2)
            
        except Exception as e:
            print(f"Error dalam handle_telegram_messages: {e}")
            time.sleep(5)

def get_latest_telegram_message():
    """Ambil pesan terbaru dari Telegram dan reset"""
    global latest_telegram_message
    with telegram_lock:
        message = latest_telegram_message
        latest_telegram_message = None
    return message

# ==== Util ====
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

# ==== YOLO Functions ====
def init_yolo_model(model_path="PKM-KC.pt"):
    """Initialize YOLO model"""
    try:
        model = YOLO(model_path)
        print(f"YOLO model loaded: {model_path}")
        return model
    except Exception as e:
        print(f"Failed to load YOLO model: {e}")
        return None

def process_frame_with_yolo(frame, model, frame_width=1280, frame_height=720):
    """Process frame dengan YOLO detection"""
    if model is None:
        return frame, 0
    
    try:
        box_annotator = sv.BoxAnnotator(thickness=2, text_thickness=2, text_scale=1)
        zone_polygon = (ZONE_POLYGON * np.array([frame_width, frame_height])).astype(int)
        zone = sv.PolygonZone(polygon=zone_polygon, frame_resolution_wh=(frame_width, frame_height))
        zone_annotator = sv.PolygonZoneAnnotator(zone=zone, color=sv.Color.red(), thickness=2, text_thickness=4, text_scale=2)

        results = model(frame, agnostic_nms=True, verbose=False)[0]
        detections = sv.Detections.from_yolov8(results)
        
        labels = [
            f"{model.model.names[class_id]} {confidence:0.2f}"
            for _, confidence, class_id, _ in detections
        ]
        
        annotated_frame = box_annotator.annotate(scene=frame.copy(), detections=detections, labels=labels)
        zone.trigger(detections=detections)
        annotated_frame = zone_annotator.annotate(scene=annotated_frame)
        
        detection_count = len(detections)
        cv2.putText(annotated_frame, f"Detections: {detection_count}", 
                   (20, frame_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return annotated_frame, detection_count
        
    except Exception as e:
        print(f"YOLO processing error: {e}")
        return frame, 0

# ==== Camera Functions ====
def list_available_cameras(max_test=3):
    """Deteksi kamera yang tersedia"""
    available_cameras = []
    print("Mencari kamera yang tersedia...")
    
    for i in range(max_test):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                available_cameras.append(i)
                print(f"Kamera {i} ditemukan - Resolusi: {frame.shape[1]}x{frame.shape[0]}")
            cap.release()
        time.sleep(0.5)
    
    return available_cameras

def init_camera(index=0, width=1280, height=720, warmup_frames=8, camera_name="Kamera"):
    """Initialize kamera dengan konfigurasi"""
    cam = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cam.isOpened():
        raise RuntimeError(f"{camera_name} {index} tidak dapat dibuka.")
    
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cam.set(cv2.CAP_PROP_FPS, 30)
    
    for _ in range(warmup_frames):
        cam.read()
        time.sleep(0.05)
    
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
    
    if overlay_text:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        text = f"{overlay_text} | {ts}"
        cv2.putText(frame, text, (18, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    
    fname = f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    fpath = os.path.join(save_dir, fname)
    ok = cv2.imwrite(fpath, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    
    if not ok:
        raise RuntimeError("Gagal menyimpan gambar.")
    
    return fpath

# ==== YOLO Camera Handler ====
def yolo_camera_handler(camera_index, yolo_model):
    """Handler khusus untuk kamera YOLO"""
    print(f"YOLO Camera Handler started for camera {camera_index}")
    
    try:
        yolo_cam = init_camera(camera_index, camera_name="YOLO Camera")
        
        while True:
            try:
                line = read_serial_data()
                
                if line and "button" in line.upper():
                    print("Sinyal button diterima! Mengambil gambar dengan YOLO...")
                    
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
                
                time.sleep(0.1)
                
            except Exception as e:
                print(f"Error in YOLO handler: {e}")
                time.sleep(1)
                
    except Exception as e:
        print(f"Failed to initialize YOLO camera: {e}")

# ==== Telegram Functions ====
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
                r = requests.post(url, data=data, files=files, timeout=20)
            
            if r.status_code == 200:
                print(f"Foto terkirim: {os.path.basename(img_path)}")
                success_count += 1
        except Exception as e:
            print(f"Error kirim foto: {e}")
    
    return success_count > 0

def send_telegram_buttons(bot_token: str, chat_id: str, resi_code: str, barang: str, harga: str):
    """Kirim button keyboard yang mengirim pesan /kisi_1 atau /kisi_2"""
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    
    keyboard = {
        "keyboard": [
            [
                {"text": "📦 KISI 1 - Kirim /kisi_1"},
                {"text": "📦 KISI 2 - Kirim /kisi_2"}
            ]
        ],
        "resize_keyboard": True,
        "one_time_keyboard": True
    }
    message_text = f"**PILIH KISI UNTUK:**\n Resi: `{resi_code}`\n Barang: {barang}\n Harga: Rp {format_rupiah(harga)}\n\nKlik button di bawah untuk mengirim perintah:"
    
    try:
        data = {
            "chat_id": chat_id,
            "text": message_text,
            "parse_mode": "Markdown",
            "reply_markup": json.dumps(keyboard)
        }
        
        response = requests.post(url, json=data, timeout=10)
        
        if response.status_code == 200:
            print("🔘 Button keyboard berhasil dikirim!")
            return True
        else:
            print("Gagal kirim button:", response.status_code)
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
        response = requests.get(url, timeout=10)
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

        while time.time() - start < timeout:
            data = read_serial_data()
            if data and "button" in data.lower():
                print("Sinyal button diterima dari Arduino!")
                return True
            time.sleep(0.5)
            
        print("Timeout - Tidak ada sinyal 'button' dari Arduino")
        return False
    except Exception as e:
        print("Gagal baca dari NODEMCU:", e)
        return False

if __name__ == "__main__":
    conn = cursor = None
    main_camera = None
    yolo_model = None
    
    try:
        if not test_telegram_connection():
            print("Tidak bisa melanjutkan, bot Telegram tidak bisa diakses")
            exit(1)
        
        # Mulai thread untuk handle pesan Telegram - PENTING!
        telegram_thread = threading.Thread(target=handle_telegram_messages, daemon=True)
        telegram_thread.start()
        print("Thread Telegram message handler started")
        
        # Load YOLO model
        yolo_model = init_yolo_model("PKM-KC.pt")
        
        # Deteksi kamera yang tersedia
        available_cams = list_available_cameras()
        if not available_cams:
            print("Tidak ada kamera yang terdeteksi!")
            exit(1)
        
        print(f"Kamera yang tersedia: {available_cams}")
        
        # Inisialisasi kamera utama
        main_camera = init_camera(available_cams[0], camera_name="Main Camera")
        
        # Mulai thread untuk kamera YOLO
        if len(available_cams) > 1 and yolo_model is not None:
            yolo_thread = threading.Thread(
                target=yolo_camera_handler, 
                args=(available_cams[1], yolo_model),
                daemon=True
            )
            yolo_thread.start()
            print("Thread YOLO camera handler started")
            
        conn, cursor = get_db_cursor()
        print("Koneksi PostgreSQL OK.")
        print("Sistem siap!")
        print("Silakan scan QR Code... (CTRL+C untuk berhenti)")
        
        while True:
            data = input().strip()
            if not data: 
                continue
            print("🔍 Hasil Scan:", data)

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
                    
                    print(f"Gambar tersimpan: {img_path}")
                    
                    photo_success = send_telegram_photos(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, [img_path], caption=caption)
                    
                    if photo_success:
                        print("Foto berhasil terkirim ke Telegram.")
                        
                        if is_cod:
                            print("Paket COD - Menunggu sinyal button dari Arduino...")
                            
                            if tunggu_perintah_dari_arduino():
                                print("Mengirim button pilihan ke Telegram...")
                                button_success = send_telegram_buttons(
                                    TELEGRAM_BOT_TOKEN, 
                                    TELEGRAM_CHAT_ID, 
                                    resi_val or data, 
                                    barang_val or "-",
                                    harga_display
                                )
                                
                                if button_success:
                                    print("Button berhasil dikirim. User bisa klik /kisi_1 atau /kisi_2")
                                    print("Pesan /kisi_1 atau /kisi_2 akan otomatis dikirim ke Arduino")
                                else:
                                    print("Gagal kirim button")
                            else:
                                print("Tidak ada sinyal button dari Arduino")
                        else:
                            print("Paket NON COD - Tidak perlu kirim button")
                    
                    else:
                        print("Gagal kirim foto")
                    
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
        
        close_serial_connection()