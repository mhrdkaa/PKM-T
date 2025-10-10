import os, time, requests, cv2, psycopg2
from datetime import datetime
from decimal import Decimal, InvalidOperation
import serial
import json
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

IOT_PORT = "COM3"
IOT_BAUD = 115200
POSTGRES_HOST = os.getenv("POSTGRES_HOST")
POSTGRES_USER = os.getenv("POSTGRES_USER")
POSTGRES_PASS = os.getenv("POSTGRES_PASS")
POSTGRES_DB   = os.getenv("POSTGRES_DB")
POSTGRES_PORT = os.getenv("POSTGRES_PORT")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "7607441123:AAE9YQU2-gFXAzWB4nnXNGpOYJyvKE41ZNY")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "5558164298")

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

# ==== Telegram Functions ====
def send_telegram_photo(bot_token: str, chat_id: str, image_path: str, caption: str = "") -> bool:
    """Kirim gambar saja (seperti sebelumnya)"""
    url = f"https://api.telegram.org/bot{bot_token}/sendPhoto"
    try:
        with open(image_path, "rb") as f:
            files = {"photo": f}
            data = {"chat_id": chat_id, "caption": caption}
            r = requests.post(url, data=data, files=files, timeout=20)
        if r.status_code == 200:
            print("Foto terkirim ke Telegram.")
            return True
        else:
            print("Gagal kirim foto:", r.status_code, r.text)
            return False
    except Exception as e:
        print("Error kirim foto:", e)
        return False

def send_telegram_buttons(bot_token: str, chat_id: str, resi_code: str, barang: str, harga: str):
    """Kirim button SETELAH gambar"""
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    keyboard = {
        "inline_keyboard": [
            [
                {"text": "KISI 1", "callback_data": f"kisi_1_{resi_code}"},
                {"text": "KISI 2", "callback_data": f"kisi_2_{resi_code}"}
            ]
        ]
    }
    
    message_text = f"**PILIH KISI UNTUK:**\n Resi: `{resi_code}`\n Barang: {barang}\n Harga: Rp {format_rupiah(harga)}"
    
    try:
        data = {
            "chat_id": chat_id,
            "text": message_text,
            "parse_mode": "Markdown",
            "reply_markup": json.dumps(keyboard)
        }
        
        response = requests.post(url, json=data, timeout=10)
        
        if response.status_code == 200:
            print("Button aksi berhasil dikirim!")
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

def init_camera(index=0, width=1280, height=720, warmup_frames=8):
    cam = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cam.isOpened():
        raise RuntimeError("Kamera tidak dapat dibuka. Cek index/driver/permission.")
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    for _ in range(warmup_frames):
        cam.read(); time.sleep(0.05)
    return cam

def capture_frame(cam, save_dir="captures", prefix="capture", overlay_text=None):
    os.makedirs(save_dir, exist_ok=True)
    ok, frame = cam.read()
    if not ok or frame is None: 
        raise RuntimeError("Gagal membaca frame dari kamera.")
    
    if overlay_text:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        text = f"{overlay_text} | {ts}"
        cv2.putText(frame, text, (18, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
    
    fname = f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    fpath = os.path.join(save_dir, fname)
    ok = cv2.imwrite(fpath, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok: 
        raise RuntimeError("Gagal menyimpan gambar.")
    return fpath
def test_telegram_connection():
    """Test koneksi ke Telegram"""
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getMe"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            print("Koneksi Telegram BOT OK")
            return True
        else:
            print(f"Telegram Bot Error: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"Error koneksi Telegram: {e}")
        return False

def tunggu_perintah_dari_arduino(timeout=60):
    try:
        ser = serial.Serial("COM8", 115200, timeout=2)  
        ser.flushInput()

        print("Menunggu sinyal 'button' dari nodemcu...")
        start = time.time()

        while time.time() - start < timeout:
            if ser.in_waiting:
                line = ser.readline().decode().strip()
                print(f"Dari nodemcu: {line}")
                if "button" in line.lower():
                    return True
        return False
    except Exception as e:
        print("Gagal konek ke NODEMCU:", e)
        return False
if __name__ == "__main__":
    conn = cursor = cam = None
    try:
        if not test_telegram_connection():
            print("Tidak bisa melanjutkan, bot Telegram tidak bisa diakses")
            exit(1)
            
        conn, cursor = get_db_cursor()
        print("Koneksi PostgreSQL OK.")
        cam = init_camera(index=0)
        print("Kamera siap.")
        print("Silakan scan QR Code... (CTRL+C untuk berhenti)")
        
        while True:
            data = input().strip()
            if not data: 
                continue
            print("Hasil Scan:", data)

            row = get_resi_detail(cursor, data)
            found = row is not None

            if found:
                print("DEBUG row dari DB:", row)

                resi_val   = pick_first(row, ["no_resi", "resi"])
                barang_val = pick_first(row, ["nama_paket", "barang", "nama_barang"])
                harga_raw  = pick_first(row, ["harga", "harga_barang"])
                status_val = pick_first(row, ["status_paket", "status"])

                harga_display = harga_raw if harga_raw is not None else "0"
                
                cap_lines = [
                    "Resi terverifikasi",
                    f"Resi  : {resi_val or '-'}",
                    f"Paket : {barang_val or '-'}",
                    f"Harga : Rp {format_rupiah(harga_display)}",
                    f"Status: {status_val or '-'}"
                ]
                caption = "\n".join(cap_lines)

                try:
                    img_path = capture_frame(cam, save_dir="captures", prefix="resi", overlay_text=f"RESI: {resi_val or data}")
                    print("Gambar tersimpan:", img_path)
                    photo_success = send_telegram_photo(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, img_path, caption=caption)
                    button_success = False
                    if photo_success:
                        print("Foto berhasil terkirim.")
                        if tunggu_perintah_dari_arduino():
                            time.sleep(5)
                            button_success = send_telegram_buttons(
                                TELEGRAM_BOT_TOKEN, 
                                TELEGRAM_CHAT_ID, 
                                resi_val or data, 
                                barang_val or "-",
                                harga_display
                            )
                            if button_success:
                                print("Button Telegram berhasil dikirim.")
                            else:
                                print("Button gagal dikirim.")
                        else:
                            print("Tidak ada sinyal dari ESP32, skip kirim button.")

                        
                        if button_success:
                            print("Gambar + Button berhasil dikirim!")
                        else:
                            print("Gambar berhasil, tapi button gagal")
                    else:
                        print("Gagal kirim foto, skip kirim button")
                    
                except Exception as e:
                    print("Gagal proses pengiriman:", e)
            else:
                print("Resi tidak ditemukan di database")

    except KeyboardInterrupt:
        print("\nProgram dihentikan oleh pengguna.")
    except Exception as e:
        print("❌ Error utama:", e)
    finally:
        try:
            if cursor: 
                cursor.close()
            if conn: 
                conn.close()
        except Exception: 
            pass
        try:
            if cam and cam.isOpened(): 
                cam.release()
        except Exception: 
            pass