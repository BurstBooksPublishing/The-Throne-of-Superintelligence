import time, hmac, hashlib, socket, threading, json
# Config (example values)
CAPACITY_BPS = 10_000_000  # link capacity
RTT = 0.2                  # seconds
SECRET = b'shared_key'     # HMAC key

def compute_bdp(capacity_bps, rtt):
    return capacity_bps * rtt  # bits

BDP = compute_bdp(CAPACITY_BPS, RTT)

def sign_payload(payload_bytes):
    return hmac.new(SECRET, payload_bytes, hashlib.sha256).hexdigest()

def timestamped_packet(sensor_id, seq, payload_dict):
    payload_dict.update({'sensor': sensor_id, 'seq': seq, 'ts': time.time()})
    payload = json.dumps(payload_dict).encode('utf-8')
    sig = sign_payload(payload)
    return json.dumps({'payload': payload.decode(), 'sig': sig}).encode('utf-8')

class AdaptiveSender(threading.Thread):
    def __init__(self, dst=('127.0.0.1',9000)):
        super().__init__(daemon=True)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.dst = dst
        self.rate_bps = CAPACITY_BPS  # start full capacity
        self.last_send = time.time()
    def run(self):
        while True:
            pkt = self._next_packet()
            if not pkt: 
                time.sleep(0.01); continue
            # enforce rate (simple token bucket)
            now = time.time(); elapsed = now - self.last_send
            # send if interval satisfied
            if len(pkt)*8 <= self.rate_bps * elapsed:
                self.sock.sendto(pkt, self.dst)  # send packet
                self.last_send = now
            else:
                # drop or lower-priority handling -- graceful degradation
                pass
    def _next_packet(self):
        # produce packets from a prioritized queue (placeholder)
        return None