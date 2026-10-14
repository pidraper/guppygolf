# Quantum Golf

Mini-golf at large hbar. The ball is a Gaussian wavepacket on a 32×32 grid. The game is compiled as a quantum circuit in Guppy and each putt runs the circuit on the Selene emulator[^1]. Brightness = |ψ|², hue = phase.  The stopping point is sampled from |ψ|².

Some angry detectors guard the course, making periodic local measurements. When a mid-circuit measurement detects the packet, the hole flees to another spot.

## Install

```bash
uv sync
uv run main.py
```

Python 3.12 or 3.13 and OpenGL 3.3 required.

## Controls

The mouse location controls aiming and power. Scroll to squeeze the packet
along one axis, click to putt and to measure early. Currently two holes are
implemented. Max 10 strokes/hole. `R` to reset.

[^1]: Guppy and Selene are Quantinuum's quantum programming language and emulator.
