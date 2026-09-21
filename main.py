import sys

import pyglet
from config import DEFAULT
from game.loop import Loop
from render.window import GolfWindow


def main():





    sys.setswitchinterval(0.0005)
    cfg = DEFAULT
    loop = Loop(cfg)
    win = GolfWindow(cfg, loop)
    pyglet.app.run()


if __name__ == "__main__":
    main()
