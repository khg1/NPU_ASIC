import numpy as np
import os

i = np.arange(256, dtype=np.uint8)
x = i.view(np.int8).astype(np.float64)

gelu = x * 0.5 * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))

lut = np.clip(np.round(gelu*256), -32768, 32767).astype(np.int16)
unsigned_lut = lut.view(np.uint16)

os.makedirs("lut", exist_ok=True)

with open("./lut/gelu_q88.hex", "w") as f:
    for value in unsigned_lut:
        f.write(f"{value & 0xFFFF:04x}\n")

# Synthesizable form: raw comma-separated array-literal entries, `include`-d
# into an unpacked array initializer (e.g. `... = '{ `include "gelu_q88.svh" };`)
# so the LUT content is a compile-time constant instead of a $readmemh file load.
with open("./lut/gelu_q88.svh", "w") as f:
    entries = [f"16'sh{value & 0xFFFF:04x}" for value in unsigned_lut]
    f.write(",\n".join(entries))
    f.write("\n")
