import asyncio
# Parameters (adjust for scenario)
N_TILES = 64
bytes_per_step = 1e9    # bytes moved per training step
flops_per_step = 1e12   # FLOPs per training step
tile_peak_flops = 1e10  # per-tile peak FLOP/s
tile_bandwidth = 1e11   # per-tile bytes/s

async def tile_work(i):
    # compute time based on FLOPs per tile
    compute_time = (flops_per_step / N_TILES) / tile_peak_flops
    # transfer time based on bytes per tile
    transfer_time = (bytes_per_step / N_TILES) / tile_bandwidth
    # simple pipelined model: start compute, overlap transfer where possible
    await asyncio.sleep(min(compute_time, transfer_time))  # overlapping portion
    # remaining serialized portion
    await asyncio.sleep(abs(compute_time - transfer_time))
    return (i, compute_time, transfer_time)

async def main():
    tasks = [tile_work(i) for i in range(N_TILES)]
    results = await asyncio.gather(*tasks)
    total_time = max(c+ t for (_, c, t) in results)  # worst-case per-tile
    print(f"Simulated step time: {total_time:.6f} s")

if __name__ == "__main__":
    asyncio.run(main())
# Adjust N_TILES, tile_bandwidth to explore trade-offs.  # end of script