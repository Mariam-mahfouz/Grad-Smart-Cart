"""
نظام الوزن للسوبر ماركت المصري
يدعم:
1. وزن افتراضي لكل منتج
2. قراءة وزن فعلي من ميزان Raspberry Pi (HX711)
"""

import time

# =============================================
# قاموس الأوزان الافتراضية لكل منتج (بالجرام)
# =============================================
PRODUCT_DEFAULT_WEIGHTS = {
    'pepsi':       330.0,   # علبة 330ml
    'cola':        330.0,
    'v cola':      250.0,
    'oreo':        137.0,   # علبة أوريو
    'biskrem':     100.0,
    'chocolate':    80.0,
    'cadbury':      90.0,
    'redbull':     250.0,   # علبة ريد بول
    'tiger':       250.0,
    'lifebuoy':    170.0,   # صابونة
    'milk':       1000.0,   # كرتونة لتر
    'juhayna':    1000.0,
    'nescafe':     200.0,   # برطمان
    'coffee':      200.0,
    'biskrem':     100.0,
    'biscuit':     100.0,
    'juice':       250.0,
    'suntop':      250.0,
    'shampoo':     400.0,
    'pantene':     400.0,
    'herbal':      400.0,
    'chips':        70.0,   # كيس شيبس
    'noodles':      75.0,
    'indomie':      75.0,
    'supermi':      75.0,
    'cheese':      200.0,
    'deodorant':   150.0,
    'nivea':       150.0,
    'tuna':        185.0,   # علبة تونة
    'beans':       400.0,
    'california':  400.0,
    'water':       600.0,
    'soda':        330.0,
    'toffifee':    125.0,
    'maxtella':    400.0,
    'zabado':       50.0,
    'big':         100.0,
    'fine':         80.0,
    'freska':      250.0,
    'hohos':        65.0,
    'plyms':       150.0,
    'rhodes':      200.0,
    'pyrosol':     300.0,
}

DEFAULT_WEIGHT_IF_UNKNOWN = 200.0  # وزن افتراضي لأي منتج مش موجود في القاموس


def get_default_weight(product_name: str) -> float:
    """إرجاع الوزن الافتراضي للمنتج"""
    name_lower = str(product_name).lower()
    for key, weight in PRODUCT_DEFAULT_WEIGHTS.items():
        if key in name_lower:
            return weight
    return DEFAULT_WEIGHT_IF_UNKNOWN


# =============================================
# كلاس الميزان (Raspberry Pi + HX711)
# =============================================
class ScaleReader:
    """
    قراءة الوزن الفعلي من ميزان HX711 على Raspberry Pi.
    لو مش متوصل بـ Pi، يرجع None وتلقائياً يستخدم الوزن الافتراضي.
    """

    def __init__(self):
        self.scale = None
        self.is_connected = False
        self.reference_unit = 1  # هتعدله بعد المعايرة
        self._try_connect()

    def _try_connect(self):
        """محاولة الاتصال بالميزان"""
        try:
            # محاولة استيراد مكتبة HX711
            from hx711 import HX711  # pip install hx711-rpi-py

            # أرقام الـ GPIO pins - عدّلهم حسب توصيلك
            DOUT_PIN = 5
            SCK_PIN  = 6

            self.scale = HX711(DOUT_PIN, SCK_PIN)
            self.scale.set_reading_format("MSB", "MSB")
            self.scale.set_reference_unit(self.reference_unit)
            self.scale.reset()
            self.scale.tare()

            self.is_connected = True
            print("✅ Scale connected successfully (HX711 on Raspberry Pi)")

        except ImportError:
            print("ℹ️  hx711 library not found → using default weights")
            self.is_connected = False
        except Exception as e:
            print(f"ℹ️  Scale not connected: {e} → using default weights")
            self.is_connected = False

    def read_weight(self, samples: int = 5) -> float | None:
        """
        قراءة الوزن الفعلي.
        Returns: الوزن بالجرام، أو None لو مش متوصل.
        """
        if not self.is_connected or self.scale is None:
            return None

        try:
            readings = []
            for _ in range(samples):
                val = self.scale.get_weight(1)
                if val > 0:
                    readings.append(val)
                time.sleep(0.05)

            if not readings:
                return None

            weight = sum(readings) / len(readings)
            return round(weight, 1)

        except Exception as e:
            print(f"Scale read error: {e}")
            return None

    def tare(self):
        """ضبط الصفر"""
        if self.is_connected and self.scale:
            try:
                self.scale.tare()
                print("⚖️  Scale tared (zero reset)")
            except Exception as e:
                print(f"Tare error: {e}")

    def set_reference_unit(self, unit: float):
        """معايرة الميزان"""
        self.reference_unit = unit
        if self.is_connected and self.scale:
            self.scale.set_reference_unit(unit)
            print(f"⚖️  Reference unit set to {unit}")


# =============================================
# دالة مساعدة: تحديد الوزن النهائي للمنتج
# =============================================
def resolve_weight(product_name: str, scale: ScaleReader) -> tuple[float, str]:
    """
    تحديد الوزن: يحاول يقرأ من الميزان الفعلي أولاً،
    لو فشل يرجع الوزن الافتراضي.

    Returns: (weight_in_grams, source_label)
    """
    actual = scale.read_weight()
    if actual is not None and actual > 5:   # تجاهل قراءات أقل من 5 جرام (ضوضاء)
        return actual, "actual"
    else:
        default = get_default_weight(product_name)
        return default, "default"