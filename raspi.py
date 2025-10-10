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

# Configuration for Raspberry Pi
IOT_PORT = "/dev/ttyUSB0"  # Raspberry Pi serial port
IOT_BAUD = 115200
POSTGRES_HOST = os.getenv("POSTGRES_HOST")
POSTGRES_USER = os.getenv("POSTGRES_USER")
POSTGRES_PASS = os.getenv("POSTGRES_PASS")
POSTGRES_DB   = os.getenv("POSTGRES_DB")
POSTGRES_PORT = os.getenv("POSTGRES_PORT")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID")

# Global variable untuk callback handler
callback_lock = threading.Lock()

# Global serial connection dengan lock
serial_lock = threading.Lock()
ser_instance = None

# Global YOLO model
yolo_model = None

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
            print(f"RAW SERIAL DATA: '{data_buffer}'")
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

# ==== Telegram Callback Handler ====
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
                        
                        # Handle callback query (inline button click)
                        if "callback_query" in update:
                            callback_data = update["callback_query"]["data"]
                            message_id = update["callback_query"]["message"]["message_id"]
                            chat_id = update["callback_query"]["message"]["chat"]["id"]
                            
                            print(f"CALLBACK DITERIMA: {callback_data}")
                            
                            # Answer the callback query (remove loading)
                            answer_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/answerCallbackQuery"
                            answer_data = {
                                "callback_query_id": update["callback_query"]["id"],
                                "text": f"Pilihan diterima: {callback_data}",
                                "show_alert": False
                            }
                            requests.post(answer_url, json=answer_data, verify=False)
                            
                            # Update message text to show selection
                            edit_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/editMessageText"
                            
                            if callback_data.startswith("kisi_1"):
                                new_text = "KISI 1 DIPILIH\nMengirim perintah ke sistem..."
                                kisi_command = "kisi_1"
                            elif callback_data.startswith("kisi_2"):
                                new_text = "KISI 2 DIPILIH\nMengirim perintah ke sistem..."
                                kisi_command = "kisi_2"
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
        if original_width > 640:  # Downscale jika resolusi terlalu besar
            frame = cv2.resize(frame, (640, 480))
            frame_width, frame_height = 640, 480
        
        # Initialize annotators
        box_annotator = sv.BoxAnnotator(
            thickness=1,  # Kurangi thickness
            text_scale=0.5  # Kurangi text scale
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
                       imgsz=320,  # Ukuran input lebih kecil
                       conf=0.5,   # Confidence threshold
                       half=False   # Non-aktifkan half precision di Raspberry Pi
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
        # Fallback: return original frame dengan watermark
        cv2.putText(frame, "YOLO Error - Original Frame", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        return frame, 0

# ==== Camera Functions ====
def list_available_cameras(max_test=3):
    """Deteksi kamera yang tersedia"""
    available_cameras = []
    print("Mencari kamera yang tersedia...")
    
    for i in range(max_test):
        # Auto-detect backend untuk Raspberry Pi
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
    # Auto-detect backend untuk Raspberry Pi
    cam = cv2.VideoCapture(index)
    if not cam.isOpened():
        raise RuntimeError(f"{camera_name} {index} tidak dapat dibuka.")
    
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    # Raspberry Pi optimization
    cam.set(cv2.CAP_PROP_FPS, 15)  # Turunkan FPS untuk Raspberry Pi
    cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer
    
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
    
    # Jika yolo_cam adalah kamera yang sama dengan main camera
    # kita perlu memastikan frame yang di-read fresh
    ok, frame = yolo_cam.read()
    if not ok or frame is None:
        raise RuntimeError("Gagal membaca frame dari kamera YOLO.")
    
    print(f"Capturing YOLO frame - Brightness: {frame.mean():.1f}")
    
    # Process dengan YOLO
    processed_frame, detection_count = process_frame_with_yolo(frame, yolo_model)
    
    fname = f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    fpath = os.path.join(save_dir, fname)
    ok = cv2.imwrite(fpath, processed_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])  # Kurangi quality
    
    if not ok:
        raise RuntimeError("Gagal menyimpan gambar YOLO.")
    
    print(f"YOLO frame saved - Detections: {detection_count}")
    return fpath, detection_count

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
    yolo_camera = None
    
    try:
        print("Starting application on Raspberry Pi...")
        
        if not test_telegram_connection():
            print("Tidak bisa melanjutkan, bot Telegram tidak bisa diakses")
            exit(1)
        
        # Load YOLO model di AWAL
        print("Loading YOLO model...")
        yolo_model = init_yolo_model("PKM-KC.pt")
        
        if yolo_model is None:
            print("YOLO model gagal di-load, sistem tetap berjalan tanpa YOLO")
        
        # Mulai thread untuk handle callback Telegram
        callback_thread = threading.Thread(target=handle_telegram_callbacks, daemon=True)
        callback_thread.start()
        print("Thread callback handler started")
        
        # Deteksi kamera yang tersedia
        available_cams = list_available_cameras()
        if not available_cams:
            print("Tidak ada kamera yang terdeteksi!")
            exit(1)
        
        print(f"Kamera yang tersedia: {available_cams}")
        
        # Inisialisasi kamera utama (untuk QR code)
        main_camera = init_camera(available_cams[0], camera_name="Main Camera")
        
        # Handle single camera scenario untuk Raspberry Pi
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
                
                # Caption untuk gambar utama
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
                    # 1. Capture dan kirim gambar utama dulu
                    img_path = capture_normal_frame(
                        main_camera,
                        save_dir="captures",
                        prefix="resi",
                        overlay_text=f"RESI: {resi_val or data}"
                    )
                    
                    print(f"Gambar utama tersimpan: {img_path}")
                    
                    photo_success = send_telegram_photos(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, [img_path], caption=caption)
                    
                    if photo_success:
                        print("Foto utama berhasil terkirim ke Telegram.")
                        
                        # 2. Jika COD, proses YOLO + Button
                        if is_cod:
                            print("Paket COD - Menunggu sinyal button dari Arduino...")
                            
                            if tunggu_perintah_dari_arduino():
                                print("Mengambil gambar YOLO...")
                                
                                # 3. Capture gambar YOLO (langsung ketika dapat sinyal button)
                                if yolo_model is not None:
                                    yolo_path, detection_count = capture_yolo_frame(yolo_camera, yolo_model)
                                    print(f"Gambar YOLO tersimpan: {yolo_path}")
                                    
                                    # 4. Kirim gambar YOLO
                                    yolo_caption = f"YOLO DETECTION\nDetections: {detection_count}\nTime: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                                    yolo_success = send_telegram_photos(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, [yolo_path], yolo_caption)
                                    
                                    if yolo_success:
                                        print("Gambar YOLO berhasil dikirim")
                                        
                                        # 5. Kirim button setelah gambar YOLO
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
                                    # Langsung kirim button jika YOLO tidak tersedia
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