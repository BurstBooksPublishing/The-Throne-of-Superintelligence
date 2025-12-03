import hmac, hashlib, time
KEY = b'shared_secret'  # placeholder for attestation verification key

def verify_token(token, message):
    # placeholder: replace with TEE attestation verification
    mac = hmac.new(KEY, message, hashlib.sha256).hexdigest()
    return mac == token

def read_sensors():
    # read ADC/GPIO in real system
    return {'v':12.0, 'i':0.8, 'temp':45.0, 'confidence':0.92}

def power_gate(state):
    # control MOSFET gate pin; here a print simulates action
    print('POWER_GATE ->', 'ENABLED' if state else 'DISABLED')

P_MAX = 20.0  # watt
C_TH = 0.8

while True:
    cmd = receive_command()            # network or bus message (blocking)
    token = cmd.pop('token')           # attestation token (signed)
    message = serialize(cmd['intent']) # canonical serialization

    if not verify_token(token, message):
        power_gate(False); log('bad_attestation'); continue

    s = read_sensors()
    p = s['v'] * s['i']

    # Atomic safety predicates
    temp_ok = s['temp'] < 70.0
    power_ok = p <= P_MAX
    conf_ok = s['confidence'] >= C_TH

    if temp_ok and power_ok and conf_ok:
        power_gate(True); execute(cmd['intent'])  # hand off to driver
    else:
        power_gate(False); log('safety_veto', temp_ok, power_ok, conf_ok)

    time.sleep(0.01)  # loop granularity