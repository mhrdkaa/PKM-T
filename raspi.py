import os
import time
import requests
import cv2
import psycopg2
from datetime import datetime
from decimal import Decimal, InvalidOperation
import serial
from serial import SerialException
import json
import threading
import numpy as np
from ultralytics import YOLO
import supervision as sv

# Supress insecure https warnings (karena verify=False dipakai)
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ======= CONFIG =======
IOT_PORT = os.getenv("IOT_PORT", "/dev/ttyUSB1")
IOT_BAUD = int(os.getenv("IOT_BAUD", "115200"))
SERIAL_RETRY_MAX = 5        # berapa kali mencoba reconnect sebelum menyerah sementara
SERIAL_RETRY_BACKOFF = 1.0  # detik, akan bertambah eksponensial

POSTGRES_HOST = os.getenv("POSTGRES_HOST")
POSTGRES_USER = os.getenv("POSTGRES_USER")
POSTGRES_PASS = os.getenv("POSTGRES_PASS")
POSTGRES_DB   = os.getenv("POSTGRES_DB")
POSTGRES_PORT = os.getenv("POSTGRES_PORT")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID")

# ======= GLOBALS =======
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

# ======= SERIAL HELPERS =======
def device_exists(path):
    try:
        return os.path.exists(path)
    except Exception:
        return False

def _open_serial_instance():
    """
    Low-level open serial: tidak menggunakan global lock (panggilan internal).
    Mengembalikan instance serial atau raise exception.
    """
    # Pastikan device ada
    if not device_exists(IOT_PORT):
        raise FileNotFoundError(f"Device serial tidak ditemukan: {IOT_PORT}")

    # Buka serial dengan timeout singkat supaya operasi tidak menggantung
    ser = serial.Serial(IOT_PORT, IOT_BAUD, timeout=1, write_timeout=1)
    # Beberapa board butuh sedikit waktu untuk siap
    time.sleep(2)
    # Reset input buffer untuk menghindari data korup tersisa
    try:
        ser.reset_input_buffer()
        ser.reset_output_buffer()
    except Exception:
        pass
    return ser

def get_serial_connection():
    """Dapatkan koneksi serial shared (dengan retry)."""
    global ser_instance
    with serial_lock:
        if ser_instance is not None:
            # cek alive
            try:
                if ser_instance.is_open:
                    return ser_instance
            except Exception:
                # kalau property error -> ulang koneksi
                try:
                    ser_instance.close()
                except Exception:
                    pass
                ser_instance = None

        # Jika belum ada, coba buka dengan retry/backoff
        attempt = 0
        backoff = SERIAL_RETRY_BACKOFF
        while attempt < SERIAL_RETRY_MAX:
            attempt += 1
            try:
                ser_instance = _open_serial_instance()
                print(f"Koneksi serial {IOT_PORT} dibuka (attempt {attempt})")
                return ser_instance
            except FileNotFoundError as e:
                # Kalau device tidak ada, tidak perlu retry panjang-panjang
                print(f"[SERIAL] Device tidak ditemukan: {e}")
                ser_instance = None
                break
            except SerialException as e:
                print(f"[SERIAL] SerialException saat membuka (attempt {attempt}): {e}")
            except OSError as e:
                print(f"[SERIAL] OSError saat membuka (attempt {attempt}): {e}")
            except Exception as e:
                print(f"[SERIAL] Error saat membuka serial (attempt {attempt}): {e}")

            # tunggu sebelum retry
            time.sleep(backoff)
            backoff = min(backoff * 2, 10.0)

        print("[SERIAL] Gagal buka koneksi serial setelah beberapa percobaan.")
        ser_instance = None
        return None

def close_serial_connection():
    """Tutup koneksi serial dengan aman"""
    global ser_instance
    with serial_lock:
        if ser_instance:
            try:
                if ser_instance.is_open:
                    ser_instance.close()
                    print("Koneksi serial ditutup")
            except Exception as e:
                print("Error saat menutup serial:", e)
            finally:
                ser_instance = None

def safe_reopen_serial():
    """Paksa close lalu coba reopen sekali (dipanggil saat terjadi error read/write)"""
    global ser_instance
    with serial_lock:
        try:
            if ser_instance:
                try:
                    ser_instance.close()
                except Exception:
                    pass
                ser_instance = None
        except Exception:
            ser_instance = None
    # coba langsung buka sekali (tidak berulang-ulang di sini)
    try:
        new_ser = _open_serial_instance()
        with serial_lock:
            ser_instance = new_ser
        print("Serial reopened successfully")
        return True
    except Exception as e:
        print("Reopen serial gagal:", e)
        return False

def read_serial_data():
    """Baca semua data yang available dari serial. Jika error, lakukan safe reopen.
       Mengembalikan string kosong jika tidak ada data atau jika serial tidak tersedia.
    """
    ser = get_serial_connection()
    if ser is None:
        return ""

    with serial_lock:
        try:
            # baca line per line bila ada (lebih robust)
            data_buffer = ""
            # cek in_waiting; jika 0 coba readline sekali (blocking dengan timeout)
            if ser.in_waiting > 0:
                chunk = ser.read(ser.in_waiting)
                try:
                    data_buffer += chunk.decode('utf-8', errors='ignore')
                except Exception:
                    pass
            else:
                # coba readline sekali (non-blocking karena timeout=1)
                try:
                    line = ser.readline()
                    if line:
                        try:
                            data_buffer += line.decode('utf-8', errors='ignore')
                        except Exception:
                            pass
                except Exception as e:
                    # readline bisa raise OSError Errno 5
                    raise

            if data_buffer:
                data_buffer = data_buffer.strip()
                print(f"RAW SERIAL DATA: '{data_buffer}'")
                return data_buffer
            return ""
        except (SerialException, OSError) as e:
            print(f"Error baca serial: {e}")
            # coba reopen sekali
            safe_reopen_serial()
            return ""
        except Exception as e:
            print(f"Unexpected error baca serial: {e}")
            return ""

def send_serial_data(data, retry_on_fail=True):
    """Kirim data melalui serial connection. Return True/False."""
    ser = get_serial_connection()
    if ser is None:
        print("[SERIAL] Tidak ada koneksi serial; tidak mengirim data.")
        return False

    with serial_lock:
        try:
            payload = f"{data}\n".encode()
            ser.write(payload)
            # flush jika ada
            try:
                ser.flush()
            except Exception:
                pass
            return True
        except (SerialException, OSError) as e:
            print(f"Error kirim serial: {e}")
            if retry_on_fail:
                # coba reopen & ulangi sekali
                reopened = safe_reopen_serial()
                if reopened:
                    return send_serial_data(data, retry_on_fail=False)
            return False
        except Exception as e:
            print(f"Unexpected error kirim serial: {e}")
            return False

# ==== Telegram Callback Handler ====
def handle_telegram_callbacks():
    print("Memulai thread callback handler...")
    last_update_id = 0

    while True:
        try:
            if not TELEGRAM_BOT_TOKEN:
                print("TELEGRAM_BOT_TOKEN tidak diset. Thread callback berhenti.")
                return

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

                            # contoh mapping callback -> command serial
                            kisi_command = None
                            if callback_data.startswith("kisi_1_"):
                                new_text = "Kisi 1 DIPILIH\nMengirim perintah ke sistem..."
                                kisi_command = "R1A"
                            elif callback_data.startswith("kisi_2_"):
                                new_text = "KISI 2 DIPILIH\nMengirim perintah ke sistem..."
                                kisi_command = "R2A"
                            else:
                                # jika tidak dikenal, skip
                                continue

                            edit_data = {
                                "chat_id": chat_id,
                                "message_id": message_id,
                                "text": new_text,
                                "parse_mode": "Markdown"
                            }
                            requests.post(edit_url, json=edit_data, verify=False)

                            if kisi_command:
                                print(f"Mengirim {kisi_command} ke Arduino...")
                                ok = send_serial_data(kisi_command)
                                print("Kirim serial result:", ok)
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
    if not status_paket:
        return False
    status_normalized = str(status_paket).upper().strip()
    print(f"DEBUG is_paket_cod: input='{status_paket}', normalized='{status_normalized}', result={status_normalized == 'COD'}")
    return status_normalized == "COD"

# ==== YOLO Functions ====
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

        box_annotator = sv.BoxAnnotator(thickness=1, text_scale=0.5)
        zone_polygon = (ZONE_POLYGON * np.array([frame_width, frame_height])).astype(int)
        zone = sv.PolygonZone(polygon=zone_polygon, frame_resolution_wh=(frame_width, frame_height))
        zone_annotator = sv.PolygonZoneAnnotator(zone=zone, color=sv.Color.red(), thickness=1, text_scale=0.8)

        results = model(frame,
                        agnostic_nms=True,
                        verbose=False,
                        imgsz=320,
                        conf=0.5,
                        half=False)[0]

        detections = sv.Detections.from_yolov8(results)

        labels = [
            f"{model.model.names[class_id]} {confidence:0.2f}"
            for _, confidence, class_id, _
            in detections
        ]

        annotated_frame = box_annotator.annotate(scene=frame.copy(), detections=detections, labels=labels)
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

# ==== Camera Functions ====
def list_available_cameras(max_test=3):
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
    cam = cv2.VideoCapture(index)
    if not cam.isOpened():
        raise RuntimeError(f"{camera_name} {index} tidak dapat dibuka.")

    cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cam.set(cv2.CAP_PROP_FPS, 15)
    try:
        cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    for i in range(warmup_frames):
        ret, frame = cam.read()
        if ret:
            print(f"{camera_name} warmup {i+1}/{warmup_frames} - Brightness: {frame.mean():.1f}")
        time.sleep(0.05)

    actual_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"{camera_name} {index} siap - Resolusi: {actual_width}x{actual_height}")

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

# ==== Telegram Functions ====
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
                print(f"Foto terkirim: {os.path.basename(img_path)}")
                success_count += 1
            else:
                print(f"Gagal kirim foto: {r.status_code} - {r.text}")
        except Exception as e:
            print(f"Error kirim foto: {e}")
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
            print("Inline button berhasil dikirim!")
            return True
        else:
            print("Gagal kirim button:", response.status_code, response.text)
            return False
    except Exception as e:
        print("Error kirim button:", e)
        return False

# ==== Database helpers ====
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
            if data:
                # data bisa berisi beberapa kata; cari kata 'button'
                if "button" in data.lower():
                    print("Sinyal button diterima dari Arduino!")
                    return True
                # kadang arduino kirim single char, e.g. '1' => Anda bisa deteksi di sini jika diperlukan
                if data.strip() in ("1", "b", "B"):
                    print("Sinyal '1' atau 'b' diterima, dianggap button")
                    return True
            time.sleep(0.5)
        print("Timeout - Tidak ada sinyal 'button' dari Arduino")
        return False
    except Exception as e:
        print("Gagal baca dari NODEMCU:", e)
        # coba reopen agar next attempt lebih baik
        safe_reopen_serial()
        return False

# ==== MAIN ====
if __name__ == "__main__":
    conn = cursor = None
    main_camera = None
    yolo_camera = None

    try:
        print("Starting application on Raspberry Pi...")

        if not test_telegram_connection():
            print("Tidak bisa melanjutkan, bot Telegram tidak bisa diakses")
            exit(1)

        print("Loading YOLO model...")
        yolo_model = init_yolo_model("PKM-KC.pt")
        if yolo_model is None:
            print("YOLO model gagal di-load, sistem tetap berjalan tanpa YOLO")

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

        # Inisialisasi koneksi database
        try:
            conn, cursor = get_db_cursor()
            print("Koneksi PostgreSQL OK.")
        except Exception as e:
            print("Gagal koneksi ke PostgreSQL:", e)
            # lanjutkan tanpa DB? Tapi sebagian fitur butuh DB
            conn = cursor = None

        print("Sistem siap!")
        print("Silakan scan QR Code... (CTRL+C untuk berhenti)")

        while True:
            try:
                data = input().strip()
            except EOFError:
                # ketika input tidak tersedia (mis. dijalankan tanpa stdin), hentikan loop
                break
            except KeyboardInterrupt:
                raise

            if not data:
                continue
            print("Hasil Scan:", data)
            row = None
            if cursor:
                try:
                    row = get_resi_detail(cursor, data)
                except Exception as e:
                    print("Error query DB:", e)
                    row = None

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

                    # Kirim perintah awal (misalnya buka_1) jika memang diinginkan
                    serial_success = send_serial_data("buka_1")
                    print("serial_success:", serial_success)

                    photo_success = send_telegram_photos(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, [img_path], caption=caption)

                    if photo_success:
                        print("Foto utama berhasil terkirim ke Telegram.")

                        if is_cod:
                            print("Paket COD - Menunggu sinyal button dari Arduino...")

                            if tunggu_perintah_dari_arduino():
                                print("Mengambil gambar YOLO...")

                                if yolo_model is not None:
                                    try:
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
                                    except Exception as e:
                                        print("Error saat capture/processing YOLO:", e)
                                        # fallback: kirim button tanpa YOLO
                                        send_telegram_buttons(
                                            TELEGRAM_BOT_TOKEN,
                                            TELEGRAM_CHAT_ID,
                                            resi_val or data,
                                            barang_val or "-",
                                            harga_display
                                        )
                                else:
                                    print("YOLO model tidak tersedia, langsung kirim button")
                                    send_telegram_buttons(
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
            if cursor:
                try:
                    cursor.close()
                except Exception:
                    pass
            if conn:
                try:
                    conn.close()
                except Exception:
                    pass
        except Exception:
            pass

        try:
            if main_camera and main_camera.isOpened():
                main_camera.release()
                print("Kamera utama di-release")
            if yolo_camera and yolo_camera.isOpened() and yolo_camera != main_camera:
                yolo_camera.release()
                print("Kamera YOLO di-release")
        except Exception:
            pass

        close_serial_connection()
