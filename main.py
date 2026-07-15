import pyglet
from config import DEFAULT
from game.loop import Loop
from render.window import GolfWindow


def main():
    cfg = DEFAULT
    loop = Loop(cfg)
    win = GolfWindow(cfg, loop)
    pyglet.app.run()


if __name__ == "__main__":
    main()
