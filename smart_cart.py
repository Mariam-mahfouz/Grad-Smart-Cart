"""
EGYPTIAN MARKET - SMART CART (REAL-TIME) + FIREBASE
====================================================
نفس منطق الكود الأصلي بالظبط + Weight.py + Threading + Firebase

التحسينات:
  ✅ Thread منفصل لقراءة الكاميرا   → مفيش تأخير في الفيديو
  ✅ Thread منفصل للـ YOLO detection → الشاشة مش بتتجمد
  ✅ نفس الـ products / prices / tracking / phone filter الأصلي
  ✅ نفس نظام الوزن من Weight.py
  ✅ Firebase integration for cart sync and orders
"""

import cv2
import numpy as np
import time
import threading
import math
import os
from datetime import datetime
from ultralytics import YOLO 
# =============================================
# Firebase Imports
# =============================================
import firebase_admin
from firebase_admin import credentials, db

# =============================================
# استيراد نظام الوزن
# =============================================
from Weight import ScaleReader, get_default_weight, resolve_weight

# =============================================
# Firebase Initialization
# =============================================
try:
    cred = credentials.Certificate("serviceAccountKey.json")
    
    if not firebase_admin._apps:
        firebase_admin.initialize_app(cred, {
            "databaseURL": "https://smartcartend-default-rtdb.firebaseio.com"  # ⚠️ ضع رابط مشروعك هنا
        })
        print("✅ Firebase initialized successfully")
    else:
        print("ℹ️ Firebase already initialized")
except Exception as e:
    print(f"⚠️ Firebase initialization error: {e}")
    print("Continuing without Firebase...")

# ══════════════════════════════════════════════
# Thread 1 : قارئ الكاميرا
# ══════════════════════════════════════════════
class CameraReader(threading.Thread):
    """
    Thread منفصل يقرأ الكاميرا باستمرار
    ويحتفظ بأحدث إطار فقط — بدون buffering
    """
    def __init__(self, src=0, width=640, height=480):
        super().__init__(daemon=True)
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # ← أهم سطر للـ real-time
        self.frame  = None
        self.ready  = False
        self._stop  = False
        self._lock  = threading.Lock()

    def run(self):
        while not self._stop:
            ret, frame = self.cap.read()
            if ret:
                with self._lock:
                    self.frame = frame
                    self.ready = True

    def read(self):
        with self._lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self._stop = True
        self.cap.release()


# ══════════════════════════════════════════════
# Thread 2 : مشغّل الـ YOLO
# ══════════════════════════════════════════════
class DetectorThread(threading.Thread):
    """
    Thread منفصل يشغّل YOLO على أحدث إطار
    ويحدّث النتائج — Main thread يرسم فقط
    """
    def __init__(self, model, conf=0.60):
        super().__init__(daemon=True)
        self.model   = model
        self.conf    = conf
        self._frame  = None
        self._results = []          # [(cls, conf, bbox, class_name), ...]
        self._lock   = threading.Lock()
        self._event  = threading.Event()
        self._stop   = False

    def submit(self, frame):
        """أرسل إطار جديد للمعالجة"""
        with self._lock:
            self._frame = frame.copy()
        self._event.set()

    def get_results(self):
        """اجلب آخر نتائج"""
        with self._lock:
            return list(self._results)

    def run(self):
        while not self._stop:
            triggered = self._event.wait(timeout=0.5)
            if not triggered or self._stop:
                continue
            self._event.clear()

            with self._lock:
                frame = self._frame

            if frame is None:
                continue

            try:
                results = self.model(frame, conf=self.conf, verbose=False)
                boxes = []
                for r in results:
                    if r.boxes is None:
                        continue
                    for i in range(len(r.boxes.cls)):
                        cls   = int(r.boxes.cls[i].item())
                        conf  = float(r.boxes.conf[i].item())
                        bbox  = r.boxes.xyxy[i].cpu().numpy().tolist()
                        cname = self.model.names.get(cls, f"Class_{cls}") \
                                if hasattr(self.model, 'names') else f"Class_{cls}"
                        boxes.append((cls, conf, bbox, cname))
                with self._lock:
                    self._results = boxes
            except Exception as e:
                print(f"[Detector] error: {e}")

    def stop(self):
        self._stop = True
        self._event.set()


# ══════════════════════════════════════════════
# الكلاس الرئيسي — نفس المنطق الأصلي + Firebase
# ══════════════════════════════════════════════
class StrictEgyptianMarketCart:
    """عربة تسوق مصرية real-time — نفس المنطق الأصلي + Threading + Weight + Firebase"""

    def __init__(self):
        # تحميل الموديل
        self.model = self.load_egyptian_model()

        # فحص فئات الموديل
        self.model_classes = self.inspect_model_classes()

        # استخراج فئات المنتجات
        self.allowed_products = self.extract_egyptian_products()

        # فلتر الهواتف
        self.phone_misclassifications = self.create_phone_filter()

        # ── نظام الوزن ──
        self.scale = ScaleReader()

        # العربة
        self.cart  = {}
        self.total = 0.0

        # نظام التتبع
        self.object_tracks    = {}
        self.next_track_id    = 0
        self.frame_index      = 0
        self.max_track_distance = 50

        # خط الدفع
        self.checkout_line_y = 350

        # إحصائيات
        self.non_product_detections = 0
        self.phones_blocked         = 0
        self.total_frames           = 0
        self.suspicious_detections  = 0
        self.background_filtered    = 0

        # إعدادات الفلترة (نفس الأصلي)
        self.min_confidence            = 0.65
        self.min_confidence_suspicious = 0.85
        self.min_object_area           = 1500
        self.min_object_width          = 40
        self.min_object_height         = 40
        self.max_object_area           = 80000
        self.roi_margin                = 20
        self.motion_threshold          = 15

        # FPS
        self._fps        = 0.0
        self._t_last     = time.time()
        self._fps_hist   = []

        scale_st = "CONNECTED ✅" if self.scale.is_connected else "NOT CONNECTED (using defaults) ℹ️"
        print("=" * 60)
        print("EGYPTIAN MARKET - REAL-TIME DETECTION + FIREBASE")
        print("=" * 60)
        print(f"Products loaded : {len(self.allowed_products)}")
        print(f"Scale           : {scale_st}")
        print("=" * 60)

    # ══════════════════════════════════════════════
    # Firebase Functions
    # ══════════════════════════════════════════════
    def sync_cart_to_firebase(self):
        """مزامنة العربة مع Firebase"""
        try:
            ref = db.reference("cart")
            ref.set({
                "items": self.cart,
                "total": self.total,
                "updated_at": str(datetime.now()),
                "item_count": len(self.cart)
            })
            print("🔄 Cart synced to Firebase")
        except Exception as e:
            print(f"⚠️ Firebase sync error: {e}")

    def save_order_to_firebase(self):
        """حفظ الطلب في Firebase"""
        try:
            ref = db.reference("orders")
            order_id = str(int(time.time()))
            
            # حساب الوزن الإجمالي
            total_weight = sum(item['weight_g'] * item['quantity'] 
                              for item in self.cart.values())
            
            ref.child(order_id).set({
                "items": self.cart,
                "total": self.total,
                "total_weight_g": total_weight,
                "total_weight_kg": total_weight / 1000,
                "status": "created",
                "created_at": str(datetime.now()),
                "item_count": len(self.cart),
                "phones_blocked": self.phones_blocked,
                "scale_connected": self.scale.is_connected
            })
            print(f"✅ Order saved to Firebase (Order ID: {order_id})")
            return order_id
        except Exception as e:
            print(f"⚠️ Firebase save error: {e}")
            return None

    # ──────────────────────────────────────────
    # نفس دوال الأصلي بدون تغيير
    # ──────────────────────────────────────────
    def create_phone_filter(self):
        return [
            'phone','mobile','cell','smartphone','iphone','samsung',
            'nokia','huawei','oppo','xiaomi','vivo','oneplus',
            'telephone','handphone','handset','device','electronic',
            'android','ios','screen','display','touchscreen',
            'rectangle','flat','glass','camera','button',
        ]

    def load_egyptian_model(self):
        model_paths = [
            'egyptian_market_final_model.pt',
            './egyptian_market_final_model.pt',
            'C:/Users/CS/Desktop/smart_cart_supermarkrt/egyptian_market_final_model.pt',
        ]
        for path in model_paths:
            if os.path.exists(path):
                print(f"Loading model: {path}")
                m = YOLO(path)
                print("Model loaded ✅")
                return m
        print("ERROR: Model not found!")
        exit(1)

    def inspect_model_classes(self):
        if hasattr(self.model, 'names'):
            print("MODEL CLASSES:")
            print("-" * 60)
            for cid, cname in self.model.names.items():
                print(f"  Class {cid}: {cname}")
            print("-" * 60)
            return self.model.names
        return {}

    def extract_egyptian_products(self):
        if not self.model_classes:
            return {}

        egyptian_products = {
            'pepsi':7.0,      'oreo':15.0,     'biskrem':5.0,
            'redbull':50.0,   'lifebuoy':20.5, 'milk':50.0,
            'nescafe':90.0,   'chocolate':12.0,'biscuit':8.0,
            'juice':8.0,      'shampoo':35.0,  'chips':5.0,
            'noodles':2.5,    'cheese':22.0,   'coffee':40.0,
            'deodorant':15.0, 'tuna':18.0,     'beans':30.0,
            'water':3.0,      'soda':6.0,      'cola':7.0,
            'pantene':45.0,   'herbal':28.0,   'cadbury':12.0,
            'toffifee':20.0,  'maxtella':60.0, 'suntop':5.0,
            'tiger':15.0,     'v cola':6.0,    'zabado':4.5,
            'juhayna':8.0,    'nivea':15.0,    'big':10.0,
            'supermi':2.5,    'indomie':2.5,   'california':30.0,
            'fine':3.0,       'freska':10.0,   'hohos':6.0,
            'plyms':18.0,     'rhodes':22.0,   'pyrosol':100.0,
        }

        allowed = {}
        print("EXTRACTING EGYPTIAN PRODUCTS:")
        print("-" * 60)

        for cid, cname in self.model_classes.items():
            cl = str(cname).lower()
            found = False
            price = 10.0
            for key, p in egyptian_products.items():
                if key in cl:
                    found = True
                    price = p
                    break
            if found:
                susp = self.is_suspicious_product(cname)
                allowed[cid] = {
                    'id':           cid,
                    'name':         cname,
                    'clean_name':   cl,
                    'price':        price,
                    'color':        self.get_product_color(cname, susp),
                    'is_suspicious': susp,
                    'requires_high_confidence': susp,
                    'default_weight': get_default_weight(cname),  # ← Weight.py
                }
                tag = "SUSPICIOUS" if susp else "ALLOWED"
                print(f"  [{tag}] {cname} — {price} EGP | ~{allowed[cid]['default_weight']:.0f}g")
            else:
                print(f"  [IGNORED] {cname}")

        print("-" * 60)
        print(f"Total allowed: {len(allowed)}")
        return allowed

    def is_suspicious_product(self, name):
        nl = str(name).lower()
        for kw in ['lifebuoy','soap','clean','hand',
                   'device','electronic','screen','glass','rectangle','flat','box']:
            if kw in nl:
                return True
        return False

    def get_product_color(self, name, suspicious=False):
        if suspicious:
            return (0, 0, 255)
        nl = str(name).lower()
        color_map = {
            'pepsi':(255,0,0),       'cola':(200,0,0),
            'oreo':(0,0,0),          'chocolate':(101,67,33),
            'redbull':(0,100,255),   'lifebuoy':(0,255,0),
            'milk':(255,255,240),    'nescafe':(101,67,33),
            'biskrem':(255,215,0),   'juice':(255,140,0),
            'shampoo':(0,255,255),   'chips':(255,255,0),
            'noodles':(255,69,0),    'cheese':(255,255,0),
            'deodorant':(128,0,128), 'tuna':(255,0,0),
            'beans':(0,100,0),       'water':(0,191,255),
            'soda':(0,255,0),        'toffifee':(210,105,30),
            'maxtella':(139,69,19),  'tiger':(255,69,0),
            'v cola':(0,0,139),      'zabado':(255,20,147),
            'juhayna':(255,215,0),   'big':(0,0,255),
            'supermi':(255,0,0),     'fine':(255,255,255),
            'freska':(0,255,255),    'hohos':(255,0,255),
            'plyms':(139,0,0),       'rhodes':(255,228,196),
            'pyrosol':(220,20,60),
        }
        for k, c in color_map.items():
            if k in nl:
                return c
        np.random.seed(hash(name) % 1000)
        return tuple(np.random.randint(100, 256, 3).tolist())

    # ── فلترة الحجم / المنطقة / الشكل (نفس الأصلي) ──
    def is_valid_object_size(self, bbox):
        x1,y1,x2,y2 = map(int, bbox)
        w = x2-x1;  h = y2-y1
        if w < self.min_object_width or h < self.min_object_height:
            return False
        area = w*h
        if area < self.min_object_area or area > self.max_object_area:
            return False
        asp = h/w if w > 0 else 0
        return 0.3 <= asp <= 3.0

    def is_in_focus_region(self, bbox, fh):
        cy = (bbox[1]+bbox[3]) / 2
        return cy >= fh * 0.2

    def is_phone_by_name(self, name):
        nl = str(name).lower()
        return any(kw in nl for kw in self.phone_misclassifications)

    def is_phone_by_shape(self, bbox, name, conf):
        x1,y1,x2,y2 = map(int, bbox)
        w = x2-x1;  h = y2-y1
        if w < 30 or h < 30:
            return False
        asp  = h/w if w > 0 else 0
        area = w*h
        if 1.7 < asp < 2.3 and 2000 < area < 30000 \
                and self.is_suspicious_product(name) and conf > 0.8:
            print(f"Phone by shape: {name}, asp={asp:.2f}")
            return True
        return False

    # ── معالجة كشف واحد (نفس الأصلي) ──
    def process_detection(self, cls, conf, bbox, class_name, fh):
        self.total_frames += 1

        if not self.is_valid_object_size(bbox):
            self.background_filtered += 1
            return None
        if not self.is_in_focus_region(bbox, fh):
            self.background_filtered += 1
            return None
        if self.is_phone_by_name(class_name):
            self.phones_blocked += 1
            return None
        if self.is_phone_by_shape(bbox, class_name, conf):
            self.phones_blocked += 1
            return None
        if cls not in self.allowed_products:
            self.non_product_detections += 1
            return None

        pinfo = self.allowed_products[cls]
        threshold = self.min_confidence_suspicious if pinfo['is_suspicious'] \
                    else self.min_confidence
        if conf < threshold:
            if pinfo['is_suspicious']:
                self.suspicious_detections += 1
            return None

        track_id = self.assign_track_id(bbox, cls, class_name)
        crossed  = self.check_line_crossing(bbox, track_id)

        if crossed and not self.is_in_cart(track_id):
            self.add_to_cart(cls, track_id, class_name)

        return {
            'track_id':   track_id,
            'crossed':    crossed,
            'in_cart':    self.is_in_cart(track_id),
            'class_name': class_name,
            'confidence': conf,
        }

    # ── Tracking (نفس الأصلي) ──
    def assign_track_id(self, bbox, class_id, class_name):
        x1,y1,x2,y2 = map(int, bbox)
        cx = (x1+x2)//2;  cy = (y1+y2)//2
        now = time.time()

        for tid in list(self.object_tracks):
            if now - self.object_tracks[tid]['last_seen'] > 3.0:
                del self.object_tracks[tid]

        best, best_d = None, float('inf')
        for tid, tk in self.object_tracks.items():
            if tk['class_id'] == class_id:
                d = math.hypot(cx-tk['center_x'], cy-tk['center_y'])
                if d < self.max_track_distance and d < best_d:
                    best_d, best = d, tid

        if best is not None:
            self.object_tracks[best].update({
                'bbox': bbox, 'center_x': cx, 'center_y': cy,
                'last_seen': now,
                'detection_count': self.object_tracks[best].get('detection_count',0)+1,
            })
            return best

        new_id = self.next_track_id
        self.next_track_id += 1
        self.object_tracks[new_id] = {
            'class_id': class_id, 'class_name': class_name,
            'bbox': bbox, 'center_x': cx, 'center_y': cy,
            'crossed': False, 'last_seen': now,
            'color': self.get_product_color(class_name, self.is_suspicious_product(class_name)),
            'is_suspicious': self.is_suspicious_product(class_name),
            'detection_count': 1,
        }
        return new_id

    def check_line_crossing(self, bbox, track_id):
        if track_id not in self.object_tracks:
            return False
        cy = (bbox[1]+bbox[3]) / 2
        tk = self.object_tracks[track_id]
        if cy > self.checkout_line_y and not tk['crossed']:
            if tk.get('detection_count', 0) > 3:
                tk['crossed'] = True
                return True
        return False

    def is_in_cart(self, track_id):
        return any(i['track_id'] == track_id for i in self.cart.values())

    # ── إضافة للعربة + الوزن + Firebase sync ──
    def add_to_cart(self, class_id, track_id, class_name):
        if self.is_in_cart(track_id):
            return False

        if class_id in self.allowed_products:
            pinfo        = self.allowed_products[class_id]
            product_name = pinfo['name']
            price        = pinfo['price']
            susp         = pinfo['is_suspicious']
        else:
            product_name = class_name
            price        = 10.0
            susp         = False

        # تأكيد المشبوهة
        if susp:
            print(f"\nWARNING: Suspicious product → {product_name}")
            print("Press 'y' within 3s to add, any other key to ignore...")
            t0 = time.time()
            while time.time()-t0 < 3:
                k = cv2.waitKey(1) & 0xFF
                if k == ord('y'):
                    break
                elif k != 255:
                    print(f"Rejected: {product_name}")
                    return False
            else:
                print(f"Timeout: ignoring {product_name}")
                return False

        # ── قراءة الوزن من Weight.py ──
        weight_g, weight_source = resolve_weight(product_name, self.scale)

        key = f"{class_id}_{track_id}"
        self.cart[key] = {
            'name':         product_name,
            'price':        price,
            'quantity':     1,
            'total':        price,
            'track_id':     track_id,
            'class_id':     class_id,
            'time_added':   datetime.now().strftime('%H:%M:%S'),
            'is_suspicious': susp,
            'weight_g':     weight_g,
            'weight_source': weight_source,
        }
        self.total = sum(i['total'] for i in self.cart.values())

        # ✅ مزامنة العربة مع Firebase
        self.sync_cart_to_firebase()

        src_icon = "⚖️ actual" if weight_source == 'actual' else "📦 default"
        warn     = " (SUSPICIOUS)" if susp else ""
        print(f"[{self.cart[key]['time_added']}] ✅ {product_name} — "
              f"{price} EGP | {weight_g:.0f}g ({src_icon}){warn}")
        return True

    # ──────────────────────────────────────────
    # الرسم
    # ──────────────────────────────────────────
    def draw_product(self, frame, bbox, class_id, conf, track_id, crossed):
        x1,y1,x2,y2 = map(int, bbox)

        if class_id in self.allowed_products:
            pinfo  = self.allowed_products[class_id]
            name   = pinfo['name']
            color  = pinfo['color']
            price  = pinfo['price']
            susp   = pinfo['is_suspicious']
            dw     = pinfo['default_weight']
        else:
            name   = f"Product_{class_id}"
            color  = (0,255,255)
            price  = 10.0
            susp   = False
            dw     = 200.0

        box_color = (0,0,255) if susp else ((0,255,0) if crossed else color)
        cv2.rectangle(frame, (x1,y1), (x2,y2), box_color, 3 if susp else 2)

        label = name + (" (SUSP)" if susp else "")
        (tw,th),_ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        bg_y1 = max(0, y1-th-10)
        cv2.rectangle(frame, (x1,bg_y1), (x1+tw+10,y1), (0,0,0), -1)
        cv2.rectangle(frame, (x1,bg_y1), (x1+tw+10,y1), box_color, 1)
        cv2.putText(frame, label, (x1+5,y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        cv2.putText(frame, f"Conf:{conf:.2f}", (x1,y2+18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200), 1)
        status = "SUSPICIOUS" if susp else ("IN CART" if crossed else f"Track#{track_id}")
        cv2.putText(frame, status, (x1,y2+36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, box_color, 1)
        cv2.putText(frame, f"{price} EGP", (x1,y2+54),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150,150,255), 1)
        cv2.putText(frame, f"~{dw:.0f}g", (x1,y2+72),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100,255,100), 1)

        if susp:
            mx,my = (x1+x2)//2, (y1+y2)//2
            cv2.line(frame,(mx-15,my-15),(mx+15,my+15),(0,0,255),2)
            cv2.line(frame,(mx-15,my+15),(mx+15,my-15),(0,0,255),2)

    def draw_checkout_line(self, frame):
        h,w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay,(0,self.checkout_line_y),(w,h),(0,50,0),-1)
        cv2.addWeighted(overlay,0.2,frame,0.8,0,frame)
        cv2.line(frame,(0,self.checkout_line_y),(w,self.checkout_line_y),(0,255,0),3)
        cv2.putText(frame,"CHECKOUT LINE",(w//2-70,self.checkout_line_y-15),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)
        cv2.putText(frame,"CROSS TO ADD TO CART",(w//2-100,self.checkout_line_y+30),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)

    def draw_statistics(self, frame):
        active    = len([t for t in self.object_tracks.values()
                         if time.time()-t['last_seen'] < 2.0])
        scale_txt = "Scale: LIVE ⚖️" if self.scale.is_connected else "Scale: DEFAULT 📦"

        cv2.putText(frame,"EGYPTIAN MARKET - REAL-TIME + FIREBASE",
                    (10,25),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)
        cv2.putText(frame,f"FPS: {self._fps:.1f}",
                    (10,50),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,255),2)
        cv2.putText(frame,f"Cart: {len(self.cart)} items | Total: {self.total:.1f} EGP",
                    (10,75),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)
        cv2.putText(frame,f"Active: {active} | Phones blocked: {self.phones_blocked}",
                    (10,100),cv2.FONT_HERSHEY_SIMPLEX,0.6,(200,200,0),2)
        cv2.putText(frame,f"Frames: {self.total_frames}",
                    (10,125),cv2.FONT_HERSHEY_SIMPLEX,0.5,(200,200,200),1)
        cv2.putText(frame,scale_txt,
                    (10,148),cv2.FONT_HERSHEY_SIMPLEX,0.5,(100,255,100),1)
        cv2.putText(frame,"q:Quit | c:Cart | r:Reset | i:Info | p:Invoice | t:Tare",
                    (10,frame.shape[0]-10),cv2.FONT_HERSHEY_SIMPLEX,0.4,(255,255,255),1)

    # ──────────────────────────────────────────
    # الحلقة الرئيسية — مع Threading
    # ──────────────────────────────────────────
    def run(self):
        if not self.allowed_products:
            print("ERROR: No products to detect!")
            return

        # ── شغّل threads ──
        cam      = CameraReader(src=0, width=640, height=480)
        detector = DetectorThread(model=self.model, conf=0.60)
        cam.start()
        detector.start()

        print("\nStarting in 2 seconds...")
        time.sleep(2)

        # انتظر أول إطار
        while not cam.ready:
            time.sleep(0.05)

        last_submit = 0.0   # وقت آخر إطار أُرسل للـ detector

        while True:
            frame = cam.read()
            if frame is None:
                continue

            # ── أرسل للـ detector كل 30ms (مش كل إطار) ──
            now = time.time()
            if now - last_submit > 0.03:
                detector.submit(frame)
                last_submit = now

            # ── حساب FPS ──
            self._fps_hist.append(now)
            self._fps_hist = [t for t in self._fps_hist if now-t < 1.0]
            self._fps = len(self._fps_hist)

            # ── رسم ──
            display = frame.copy()
            self.draw_checkout_line(display)

            fh = frame.shape[0]
            for cls, conf, bbox, class_name in detector.get_results():
                result = self.process_detection(cls, conf, bbox, class_name, fh)
                if result:
                    self.draw_product(display, bbox, cls, conf,
                                      result['track_id'], result['in_cart'])

            self.draw_statistics(display)
            cv2.imshow('EGYPTIAN MARKET - REAL-TIME + FIREBASE', display)

            # ── مفاتيح ──
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                self.show_cart()
            elif key == ord('r'):
                self.reset_cart()
            elif key == ord('i'):
                self.show_system_info()
            elif key == ord('s'):
                self.save_screenshot(display)
            elif key == ord('p'):
                self.show_invoice(save_to_file=True)
                # ✅ حفظ الطلب في Firebase
                order_id = self.save_order_to_firebase()
                if order_id:
                    print(f"🧾 Order #{order_id} saved to Firebase")
            elif key == ord('t'):
                self.scale.tare()

        # ── إيقاف ──
        detector.stop()
        cam.stop()
        cv2.destroyAllWindows()
        self.show_final_report()

    # ──────────────────────────────────────────
    # نفس دوال العرض الأصلية
    # ──────────────────────────────────────────
    def show_cart(self):
        print("\n" + "="*60)
        print("YOUR CART")
        print("="*60)
        if not self.cart:
            print("Cart is empty")
        else:
            tot_items = tot_price = 0
            for item in self.cart.values():
                src  = "⚖️" if item['weight_source']=='actual' else "📦"
                warn = " (SUSPICIOUS)" if item['is_suspicious'] else ""
                print(f"• {item['name']:<20} x{item['quantity']:2} = "
                      f"{item['total']:5.1f} EGP | "
                      f"{item['weight_g']:.0f}g{src} "
                      f"({item['time_added']}){warn}")
                tot_items += item['quantity']
                tot_price += item['total']
            print("-"*60)
            print(f"Total: {tot_items} items | {tot_price:.1f} EGP")
        print("="*60)

    def reset_cart(self):
        self.cart          = {}
        self.total         = 0.0
        self.object_tracks = {}
        self.next_track_id = 0
        # ✅ مزامنة العربة الفارغة مع Firebase
        self.sync_cart_to_firebase()
        print("Cart reset ✅")

    def save_screenshot(self, frame):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        cv2.imwrite(f"screenshot_{ts}.jpg", frame)
        print(f"Screenshot saved: screenshot_{ts}.jpg")

    def show_system_info(self):
        print("\n" + "="*60)
        print("SYSTEM INFO")
        print("="*60)
        print(f"Products    : {len(self.allowed_products)}")
        print(f"Cart items  : {len(self.cart)}")
        print(f"Total       : {self.total:.1f} EGP")
        print(f"Phones blk  : {self.phones_blocked}")
        print(f"Frames      : {self.total_frames}")
        print(f"FPS         : {self._fps:.1f}")
        print(f"Scale       : {'CONNECTED' if self.scale.is_connected else 'DISCONNECTED'}")
        print("="*60)

    def show_invoice(self, save_to_file=False):
        print("\n" + "="*60)
        print("         🧾 EGYPTIAN MARKET - INVOICE WITH WEIGHT")
        print("="*60)
        print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        src_note = "⚖️ Live scale" if self.scale.is_connected else "📦 Default weights"
        print(f"Weight source: {src_note}")
        print("-"*60)

        if not self.cart:
            print("No items in cart")
            print("="*60)
            return

        print(f"{'No.':<4} {'Product':<20} {'Qty':>4} {'Weight':>9} {'Price':>8} {'Total':>10}")
        print("-"*60)

        n = tot_items = tot_price = tot_weight = 0
        for item in self.cart.values():
            n += 1
            src  = "⚖️" if item['weight_source']=='actual' else "📦"
            warn = " ⚠️" if item['is_suspicious'] else ""
            print(f"{n:<4} {item['name']:<20} {item['quantity']:>4} "
                  f"{item['weight_g']:>6.0f}g{src} "
                  f"{item['price']:>7.1f}EGP {item['total']:>9.1f}EGP{warn}")
            tot_items  += item['quantity']
            tot_price  += item['total']
            tot_weight += item['weight_g'] * item['quantity']

        print("-"*60)
        print(f"TOTAL: {tot_items} items | {tot_weight:.0f}g ({tot_weight/1000:.2f}kg) | {tot_price:.1f} EGP")
        print("\nThank you for shopping! 🛍️")
        print("="*60)

        if save_to_file:
            ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
            name = f"invoice_{ts}.txt"
            with open(name,'w',encoding='utf-8') as f:
                f.write(f"EGYPTIAN MARKET INVOICE\n{datetime.now()}\n{'='*50}\n")
                for i,item in enumerate(self.cart.values(),1):
                    src = "(actual)" if item['weight_source']=='actual' else "(default)"
                    f.write(f"{i}. {item['name']} x{item['quantity']} "
                            f"| {item['weight_g']:.0f}g {src} "
                            f"| {item['price']:.1f} EGP\n")
                f.write(f"{'='*50}\n")
                f.write(f"Total: {tot_items} items | {tot_weight:.0f}g | {tot_price:.1f} EGP\n")
            print(f"📄 Invoice saved: {name}")

    def show_final_report(self):
        print("\n" + "="*60)
        print("FINAL REPORT")
        print("="*60)
        self.show_cart()
        print(f"\nPhones blocked: {self.phones_blocked}")
        print(f"Frames processed: {self.total_frames}")
        self.show_invoice(save_to_file=True)


# ══════════════════════════════════════════════
def main():
    print("\n" + "="*60)
    print("   EGYPTIAN MARKET - REAL-TIME SMART CART + FIREBASE")
    print("="*60)
    cart = StrictEgyptianMarketCart()
    cart.run()


if __name__ == "__main__":
    main()