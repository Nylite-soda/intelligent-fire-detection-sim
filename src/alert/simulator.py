import os
import datetime
from abc import ABC, abstractmethod

class AlertDispatcher(ABC):
    @abstractmethod
    def dispatch(self, zone_id, confidence, alert_type):
        pass

class MockHardwareAlert(AlertDispatcher):
    """
    Mocks the hardware alert chain (GSM/IoT). 
    A future physical GSM/IoT module can drop in by extending AlertDispatcher.
    """
    def __init__(self, log_dir="logs"):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_file = os.path.join(self.log_dir, "alerts.log")
        
    def dispatch(self, zone_id, confidence, alert_type="Active Fire"):
        timestamp = datetime.datetime.now().isoformat()
        
        # Mocks
        sms_status = "SMS_SENT"
        iot_status = "IOT_ALERT_SENT"
        
        log_entry = f"[{timestamp}] ZONE:{zone_id} | ALERT:{alert_type} | CONFIDENCE:{confidence:.2f} | {sms_status} | {iot_status}\n"
        
        # Write to log
        with open(self.log_file, "a") as f:
            f.write(log_entry)
            
        # Console Alert
        print("\n" + "="*50)
        print(" 🔥🔥🔥 CRITICAL ALERT: ACTIVE FIRE DETECTED 🔥🔥🔥")
        print("="*50)
        print(f" Zone ID    : {zone_id}")
        print(f" Confidence : {confidence*100:.1f}%")
        print(f" SMS Status : {sms_status}")
        print(f" IoT Status : {iot_status}")
        print("="*50 + "\n")
