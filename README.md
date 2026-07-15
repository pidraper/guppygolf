# Quantum Golf

Mini-golf played with a wavefunction. The ball is a Gaussian wavepacket, launched across a two-dimensional potential and evolved by the Schrödinger equation on a 16×16 grid, four qubits per axis. Each stroke compiles a split-operator circuit in Guppy, runs it once on the Selene emulator,[^1] and reads the packet back at every step. The field is domain-colored, hue carrying the phase and brightness the probability density |ψ|². Where the ball settles is sampled from |ψ|² by the Born rule.

Six detectors watch the course, their cells drawn at random each game. When a mid-circuit measurement finds the packet inside a watched cell, the hole flees to another burrow before the ball can come to rest.

## Install

```
pip install -r requirements.txt
python main.py
```

Python 3.10 through 3.13 is required, together with a display and OpenGL 3.3.[^2]

## Controls

To aim, move the mouse; the launch direction and power follow the cursor. Scroll to squeeze the packet along one axis, an area-preserving squash that trades width for height. Click to putt. Press `R` to reset.

[^1]: Guppy and Selene are Quantinuum's quantum programming language and emulator. The circuit is ordinary Python, recompiled each stroke.

[^2]: The field renders on the GPU through moderngl, so a headless machine will not run it.
