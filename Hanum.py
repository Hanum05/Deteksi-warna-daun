import cv2
import numpy as np

# --- Klasifikasi berdasarkan tabel kelayakan daun ---
def klasifikasi_warna(mean_hsv):
    h, s, v = mean_hsv

    # Hijau Cerah
    if 35 <= h <= 85 and s > 80 and v > 120:
        return "Hijau Cerah (Sangat Sehat)", "Kurang Direkomendasikan"

    # Hijau Tua / Pekat
    if 35 <= h <= 85 and s > 40 and v <= 120:
        return "Hijau Tua (Sehat)", "Kurang Direkomendasikan (kecuali diarangkan)"

    # Kuning / Kuning Kecoklatan
    if 20 <= h < 35:
        return "Kuning (Menua)", "Direkomendasikan"

    # Coklat / Daun Kering
    if 5 <= h < 20 and v < 200:
        return "Coklat / Kering", "Sangat Direkomendasikan"

    return "Tidak Terdeteksi", "-"


# --- Akses kamera ---
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Kamera tidak ditemukan!")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Mask daun (umum: hijau → kuning → coklat muda)
    lower_leaf = np.array([15, 20, 20])
    upper_leaf = np.array([95, 255, 255])

    mask = cv2.inRange(hsv, lower_leaf, upper_leaf)

    # Membersihkan noise
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    hasil_warna = "Tidak Ada Daun"
    rekomendasi = "-"

    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)

        if area > 1000:
            x, y, w, h = cv2.boundingRect(c)

            leaf_region = hsv[y:y+h, x:x+w]

            mean_color = np.mean(leaf_region.reshape(-1, 3), axis=0)

            hasil_warna, rekomendasi = klasifikasi_warna(mean_color)

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

            cv2.putText(frame, hasil_warna, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

            cv2.putText(frame, f"Kelayakan: {rekomendasi}",
                        (x, y + h + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    cv2.imshow("Deteksi Daun + Kelayakan Tinta", frame)
    cv2.imshow("Mask Daun", mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
