import cv2
import mediapipe as mp
import numpy as np
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import math
from PIL import Image, ImageDraw, ImageFont
import os

class VolumeController:
    def __init__(self):
        # ตั้งค่า MediaPipe สำหรับตรวจจับมือ
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # ตั้งค่าการควบคุมเสียง
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        self.volume = cast(interface, POINTER(IAudioEndpointVolume))
        self.volume_range = self.volume.GetVolumeRange()  # (-65.25, 0.0)
        
        # เพิ่มตัวแปรสำหรับติดตามสถานะ Mute
        self.is_muted = False
        self.last_volume = None  # เก็บค่าระดับเสียงก่อน mute
        self.mute_gesture_count = 0  # นับเฟรมที่ตรวจพบท่ากำมือ
        self.mute_threshold = 10  # จำนวนเฟรมขั้นต่ำที่ต้องตรวจพบท่ากำมือ
        
        # ตั้งค่าหน้าต่างแสดงผล
        self.window_name = 'ควบคุมเสียงด้วยมือ'
        cv2.namedWindow(self.window_name)
        
        # ตั้งค่าฟอนต์ภาษาไทย
        try:
            font_path = "C:/Windows/Fonts/tahoma.ttf"  # ใช้ฟอนต์ Tahoma ที่มีในวินโดวส์
            self.font = ImageFont.truetype(font_path, 32)
            self.small_font = ImageFont.truetype(font_path, 24)
            self.font_ok = True
        except:
            print("ไม่สามารถโหลดฟอนต์ภาษาไทยได้")
            self.font_ok = False

    def get_hand_landmarks(self, frame):
        """ตรวจจับตำแหน่งมือในภาพ"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        return results.multi_hand_landmarks
    
    def calculate_distance(self, p1, p2):
        """คำนวณระยะห่างระหว่างสองจุด"""
        return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
    
    def get_volume_percentage(self, volume_db):
        """แปลงค่าเสียงเป็นเปอร์เซ็นต์"""
        min_vol = self.volume_range[0]  # -65.25
        max_vol = self.volume_range[1]  # 0.0
        return int(((volume_db - min_vol) / (max_vol - min_vol)) * 100)
    
    def draw_volume_bar(self, frame, volume_percent):
        """วาดแถบแสดงระดับเสียง"""
        frame_height, frame_width = frame.shape[:2]
        bar_height = int(frame_height * 0.6)  # ความสูง 60% ของภาพ
        bar_width = 30  # ความกว้างแถบ
        
        # ตำแหน่งแถบ (ชิดขวา)
        x1 = frame_width - 50  # ห่างจากขอบขวา 50 พิกเซล
        y1 = int((frame_height - bar_height) / 2)
        x2 = x1 + bar_width
        y2 = y1 + bar_height
        
        # วาดกรอบแถบ
        cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 200, 200), 2)
        
        # วาดระดับเสียง
        bar_fill_height = int(bar_height * (volume_percent / 100))
        y2_fill = y2 - bar_fill_height
        cv2.rectangle(frame, (x1, y2), (x2, y2_fill), (0, 255, 0), cv2.FILLED)
        
        # แสดงตัวเลขเปอร์เซ็นต์
        cv2.putText(frame, f'{volume_percent}%', 
                   (x1 - 5, y2 + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (255, 255, 255), 2)
    
    def put_thai_text(self, frame, text, position, font_size=32):
        """แสดงข้อความภาษาไทยบนภาพ"""
        if not self.font_ok:
            # ถ้าไม่มีฟอนต์ภาษาไทย ใช้ฟอนต์เริ่มต้น
            cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (255, 255, 255), 2)
            return frame

        # แปลง OpenCV frame เป็น PIL Image
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(image)
        
        # เลือกฟอนต์ตามขนาด
        font = self.small_font if font_size < 32 else self.font
        
        # วาดข้อความ
        draw.text(position, text, font=font, fill=(255, 255, 255))
        
        # แปลงกลับเป็น OpenCV frame
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    def is_fist(self, landmarks):
        """ตรวจสอบว่าเป็นท่ากำมือหรือไม่"""
        # ตำแหน่งข้อนิ้วทั้งหมด
        finger_tips = [8, 12, 16, 20]  # ปลายนิ้วชี้, กลาง, นาง, ก้อย
        finger_mids = [6, 10, 14, 18]  # ข้อกลางนิ้วชี้, กลาง, นาง, ก้อย
        
        # ตรวจสอบว่าปลายนิ้วอยู่ต่ำกว่าข้อกลางหรือไม่ (นิ้วงอ)
        fingers_bent = all(
            landmarks.landmark[tip].y > landmarks.landmark[mid].y
            for tip, mid in zip(finger_tips, finger_mids)
        )
        
        # ตรวจสอบนิ้วโป้ง (ต้องอยู่ใกล้ฝ่ามือ)
        thumb_tip = landmarks.landmark[4]
        thumb_base = landmarks.landmark[2]
        thumb_bent = thumb_tip.x > thumb_base.x if landmarks.landmark[0].x < 0.5 else thumb_tip.x < thumb_base.x
        
        return fingers_bent and thumb_bent

    def toggle_mute(self):
        """สลับสถานะ Mute/Unmute"""
        if not self.is_muted:
            # เก็บค่าระดับเสียงปัจจุบันก่อน mute
            self.last_volume = self.volume.GetMasterVolumeLevelScalar()
            self.volume.SetMute(True, None)
            self.is_muted = True
        else:
            # คืนค่าระดับเสียงเดิม
            self.volume.SetMute(False, None)
            if self.last_volume is not None:
                self.volume.SetMasterVolumeLevelScalar(self.last_volume, None)
            self.is_muted = False

    def draw_mute_status(self, frame):
        """วาดสถานะ Mute บนภาพ"""
        if self.is_muted:
            # วาดไอคอน mute และข้อความ
            frame = self.put_thai_text(frame, "🔇 ปิดเสียง", (10, 110))
        else:
            # วาดไอคอน volume และข้อความ
            frame = self.put_thai_text(frame, "🔊 เปิดเสียง", (10, 110))

    def process_frame(self, frame):
        """ประมวลผลแต่ละเฟรมและควบคุมเสียง"""
        hand_landmarks = self.get_hand_landmarks(frame)
        
        if hand_landmarks:
            for landmarks in hand_landmarks:
                # วาดจุดและเส้นบนมือ
                self.mp_draw.draw_landmarks(
                    frame, landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                # ตรวจสอบท่ากำมือ
                if self.is_fist(landmarks):
                    self.mute_gesture_count += 1
                    if self.mute_gesture_count >= self.mute_threshold:
                        self.toggle_mute()
                        self.mute_gesture_count = 0  # รีเซ็ตตัวนับ
                else:
                    self.mute_gesture_count = 0  # รีเซ็ตตัวนับเมื่อไม่ใช่ท่ากำมือ
                
                # ถ้าไม่ได้ปิดเสียง ให้ควบคุมระดับเสียงตามปกติ
                if not self.is_muted:
                    # ใช้ปลายนิ้วชี้และนิ้วโป้งในการควบคุม
                    thumb_tip = landmarks.landmark[4]  # นิ้วโป้ง
                    index_tip = landmarks.landmark[8]  # นิ้วชี้
                    
                    # แปลงพิกัดเป็นพิกเซล
                    frame_height, frame_width = frame.shape[:2]
                    thumb_x, thumb_y = int(thumb_tip.x * frame_width), int(thumb_tip.y * frame_height)
                    index_x, index_y = int(index_tip.x * frame_width), int(index_tip.y * frame_height)
                    
                    # วาดเส้นและวงกลมระหว่างนิ้ว
                    cv2.line(frame, (thumb_x, thumb_y), (index_x, index_y), (0, 255, 0), 2)
                    cv2.circle(frame, (thumb_x, thumb_y), 10, (255, 0, 0), cv2.FILLED)
                    cv2.circle(frame, (index_x, index_y), 10, (255, 0, 0), cv2.FILLED)
                    
                    # คำนวณระยะห่างและปรับระดับเสียง
                    distance = self.calculate_distance((thumb_x, thumb_y), (index_x, index_y))
                    
                    # แปลงระยะห่างเป็นระดับเสียง (ปรับค่าตามความเหมาะสม)
                    min_distance = 20   # ระยะห่างต่ำสุด (เสียงเบาสุด)
                    max_distance = 200  # ระยะห่างสูงสุด (เสียงดังสุด)
                    
                    # ปรับให้อยู่ในช่วง 0-100
                    volume_percent = np.interp(distance, [min_distance, max_distance], [0, 100])
                    volume_percent = min(max(volume_percent, 0), 100)  # จำกัดช่วง 0-100
                    
                    # แปลงเป็นค่าระดับเสียงและตั้งค่า
                    volume_db = np.interp(volume_percent, [0, 100], [self.volume_range[0], self.volume_range[1]])
                    self.volume.SetMasterVolumeLevel(volume_db, None)
                    
                    # วาดแถบแสดงระดับเสียง
                    self.draw_volume_bar(frame, int(volume_percent))
                
                # แสดงคำแนะนำภาษาไทย
                frame = self.put_thai_text(frame, "ใช้นิ้วโป้งและนิ้วชี้ควบคุมระดับเสียง", (10, 30))
                frame = self.put_thai_text(frame, "กำมือค้างไว้เพื่อ เปิด/ปิด เสียง", (10, 70), 24)
                frame = self.put_thai_text(frame, "กด 'q' หรือ ESC เพื่อออก", (10, 150), 24)
                
                # แสดงสถานะ Mute
                self.draw_mute_status(frame)
        
        return frame

    def run(self):
        """เริ่มการทำงานของโปรแกรม"""
        cap = cv2.VideoCapture(0)
        
        print("\nวิธีออกจากโปรแกรม:")
        print("1. กด 'q' หรือ 'Q'")
        print("2. กด 'ESC'")
        print("3. กด Ctrl+C ที่หน้าต่าง Terminal")
        print("\nวิธีใช้งาน:")
        print("1. ใช้นิ้วโป้งและนิ้วชี้ควบคุมระดับเสียง")
        print("2. กำมือค้างไว้เพื่อ เปิด/ปิด เสียง")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # พลิกภาพแนวนอนเพื่อให้เป็นภาพกระจก
                frame = cv2.flip(frame, 1)
                
                # ประมวลผลภาพ
                frame = self.process_frame(frame)
                
                # แสดงผล
                cv2.imshow(self.window_name, frame)
                
                # ตรวจสอบการกดปุ่ม
                key = cv2.waitKey(1) & 0xFF
                if key in [ord('q'), ord('Q'), 27]:  # q, Q, หรือ ESC
                    print("\nกำลังปิดโปรแกรม...")
                    break
        
        except KeyboardInterrupt:
            print("\nผู้ใช้หยุดการทำงานของโปรแกรม")
        finally:
            print("กำลังปิดกล้องและหน้าต่าง...")
            cap.release()
            cv2.destroyAllWindows()
            print("ปิดโปรแกรมเรียบร้อย")

def main():
    """ฟังก์ชันหลัก"""
    print("\nกำลังเริ่มโปรแกรมควบคุมเสียงด้วยมือ...")
    print("คำแนะนำ:")
    print("1. ใช้นิ้วโป้งและนิ้วชี้ในการควบคุมระดับเสียง")
    print("2. ยิ่งแยกนิ้วห่างกัน เสียงจะยิ่งดัง")
    print("3. กด 'q' เพื่อออกจากโปรแกรม\n")
    
    controller = VolumeController()
    controller.run()

if __name__ == '__main__':
    main() 