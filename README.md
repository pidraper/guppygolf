# Quantum Golf

Mini-golf at large hbar. The ball is a Gaussian wavepacket, putted across a two-dimensional potential on a 16×16 grid. Each stroke compiles a quantum circuit in Guppy, runs it once on the Selene emulator,[^1] and reads the packet back at every step. Brightness = |ψ|², hue = phase.  The stopping point is sampled from |ψ|².

Six angry detectors guard the course, making periodic local measurements. When a mid-circuit measurement finds the packet inside a watched cell, the hole flees to another burrow.

## Install

```
pip install -r requirements.txt
python main.py
```

Python 3.10 through 3.13 is required, together with a display and OpenGL 3.3.

## Controls

The mouse location controls aiming and power. Scroll to squeeze the packet along one axis,  Click to putt and to measure early.  `R` to reset.

[^1]: Guppy and Selene are Quantinuum's quantum programming language and emulator. The circuit is ordinary Python.
