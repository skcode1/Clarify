"""Real-time test: enhance input.jpeg and save output in the same directory."""

from pathlib import Path
from clarify import enhance

INPUT = Path("input.jpeg")

if not INPUT.is_file():
    raise SystemExit(f"Input file not found: {INPUT.resolve()}")

# AI super-resolution 2x -> PNG
out_ai = enhance(INPUT, mode="ai", scale=2, output_format="png")
print(f"AI 2x (PNG):    {out_ai}")

# AI super-resolution 4x -> WebP
out_webp = enhance(INPUT, mode="ai", scale=4, output_format="webp", quality=90)
print(f"AI 4x (WebP):   {out_webp}")

# Classical enhancement -> JPEG
out_classical = enhance(INPUT, mode="classical", output_format="jpeg", quality=92)
print(f"Classical (JPEG): {out_classical}")

# Denoise -> TIFF
out_denoise = enhance(INPUT, mode="denoise", strength=1.5, output_format="tiff")
print(f"Denoise (TIFF): {out_denoise}")

# Deblur -> PNG
out_deblur = enhance(INPUT, mode="deblur", strength=1.2, output_format="png")
print(f"Deblur (PNG):   {out_deblur}")

print("\nAll outputs saved next to input.jpeg")
