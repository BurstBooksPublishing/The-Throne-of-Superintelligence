import asyncio, time, random, json
# Simple weights reflect trust per source.
WEIGHTS = {'sat':0.3, 'mesh':0.4, 'lab':0.3}
THRESH = 0.8

async def producer(name, q, mean_delay, jitter, value_fn):
    seq = 0
    while seq < 50:
        await asyncio.sleep(max(0, random.gauss(mean_delay, jitter)))
        ts = time.time()  # GNSS-like timestamp
        value = value_fn()
        # signed payload would be added here in production.
        await q.put({'src':name,'seq':seq,'ts':ts,'v':value})
        seq += 1

async def aggregator(q, out_q):
    buffer = []
    while True:
        item = await q.get()
        buffer.append(item)
        # merge condition: if oldest sample older than window, fuse.
        now = time.time()
        window = 0.2  # seconds
        if now - min(x['ts'] for x in buffer) > window:
            # simple time-aligned fusion by weighted average.
            groups = {}
            for x in buffer:
                groups.setdefault(x['src'], []).append(x)
            fused_val = sum(WEIGHTS[s]* (sum(v['v'] for v in grp)/len(grp))
                            for s,grp in groups.items())
            # confidence: inverse variance proxy
            var = max(1e-6, sum((v['v']-fused_val)**2 for v in buffer)/len(buffer))
            conf = 1.0 / (1.0 + var)
            await out_q.put({'ts':now,'val':fused_val,'conf':conf,'n':len(buffer)})
            buffer.clear()

async def controller(out_q):
    while True:
        msg = await out_q.get()
        # control decision: emit action when confidence is high.
        if msg['conf'] > THRESH:
            # action would be an API call to actuator/robotic system.
            print(json.dumps({'action':'apply','ts':msg['ts'],
                              'val':msg['val'],'conf':msg['conf']}))
        else:
            # log low-confidence for human review or experiment replanning.
            print(json.dumps({'log':'low_conf','ts':msg['ts'],
                              'val':msg['val'],'conf':msg['conf']}))

async def main():
    q = asyncio.Queue()
    out_q = asyncio.Queue()
    # producers: satellite (higher jitter), mesh (moderate), lab (low-latency).
    tasks = [
        producer('sat', q, 0.08, 0.04, lambda: random.gauss(10,2)),
        producer('mesh', q, 0.03, 0.01, lambda: random.gauss(10,1)),
        producer('lab', q, 0.01, 0.005, lambda: random.gauss(10,0.2)),
        aggregator(q, out_q),
        controller(out_q)
    ]
    await asyncio.gather(*tasks)

if __name__ == '__main__':
    asyncio.run(main())