#!/usr/bin/env python3
"""Writes a synthetic equirectangular HDR sky used to exercise the dome light.

The map contains a bright sun disc, a sky gradient and a darker ground half,
which makes environment importance sampling easy to verify: a correct renderer
puts a sharp shadow under objects lit by this map.
"""

import argparse
import math
import struct


def rgbe(r, g, b):
    v = max(r, g, b)
    if v < 1e-32:
        return (0, 0, 0, 0)
    mantissa, exponent = math.frexp(v)
    scale = mantissa * 256.0 / v
    return (
        min(255, int(r * scale)),
        min(255, int(g * scale)),
        min(255, int(b * scale)),
        exponent + 128,
    )


def sky_color(direction, sun_direction, sun_intensity):
    x, y, z = direction
    if y >= 0.0:
        horizon = (0.85, 0.90, 1.00)
        zenith = (0.16, 0.32, 0.72)
        t = y ** 0.55
        base = [h * (1.0 - t) + z_ * t for h, z_ in zip(horizon, zenith)]
        base = [c * 1.4 for c in base]
    else:
        base = [0.10, 0.09, 0.08]

    cos_angle = sum(a * b for a, b in zip(direction, sun_direction))
    sun_angular_radius = math.radians(2.5)
    if cos_angle > math.cos(sun_angular_radius):
        return [c + sun_intensity for c in base]
    # Soft glow around the sun.
    glow = max(0.0, cos_angle) ** 220 * sun_intensity * 0.02
    return [c + glow for c in base]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", nargs="?", default="test_sky.hdr")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--sun-intensity", type=float, default=400.0)
    parser.add_argument("--sun-elevation", type=float, default=38.0)
    parser.add_argument("--sun-azimuth", type=float, default=-40.0)
    args = parser.parse_args()

    elevation = math.radians(args.sun_elevation)
    azimuth = math.radians(args.sun_azimuth)
    sun_direction = (
        math.cos(elevation) * math.sin(azimuth),
        math.sin(elevation),
        -math.cos(elevation) * math.cos(azimuth),
    )

    with open(args.output, "wb") as handle:
        handle.write(b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n")
        handle.write(f"-Y {args.height} +X {args.width}\n".encode("ascii"))
        for row in range(args.height):
            v = (row + 0.5) / args.height
            theta = v * math.pi
            scanline = bytearray()
            for column in range(args.width):
                u = (column + 0.5) / args.width
                phi = (u - 0.5) * 2.0 * math.pi
                direction = (
                    math.sin(theta) * math.sin(phi),
                    math.cos(theta),
                    -math.sin(theta) * math.cos(phi),
                )
                color = sky_color(direction, sun_direction, args.sun_intensity)
                scanline += struct.pack("BBBB", *rgbe(*color))
            handle.write(bytes(scanline))
    print(f"Wrote {args.output} ({args.width}x{args.height})")


if __name__ == "__main__":
    main()
