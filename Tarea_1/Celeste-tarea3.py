from matplotlib.pyplot import grid, xlabel, ylabel, subplots, axes, show
from matplotlib.animation import FuncAnimation as anim; from matplotlib.widgets import TextBox
from numpy import linspace; from random import random; from time import sleep
with open('data1.txt', 'r') as file:
    fig, ax = subplots()
    cubic = lambda s, e, t: ((1 - t) ** 3 * s) + (3 * (1 - t) ** 2 * t * s) + (3 * (1 - t) * t ** 2 * e) + (t**3 * e)
    z = [[random() * 2, random() * 10] for i in range(5)]
    x, y = map(list, zip(*[(float(i.strip().split(',')[0]), float(i.strip().split(',')[1])) for i in file]))
    xh, = ax.plot([], [], 'r--')
    yh, = ax.plot([], [], 'r--')
    point, = ax.plot([], [], 'ro')
    ls = linspace(min(x), max(x), 100)
    line, = ax.plot([], [], 'r-')
    ax.scatter(x, y, c = 'blue', marker = 'o', alpha = .6, s = 50)
    ax.set_position((0.1, 0.25, 0.8, 0.65))
    xlabel('Eje X', fontsize = 12, )
    ylabel('Eje Y', fontsize = 12)
    grid(True, linestyle = '--', alpha = .4)
    def update(frame):
        z0, z1 = z[int(frame/20)%5], z[(int(frame/20)+1)%5]
        g = cubic(z0[0], z1[0], (frame%20)/20), cubic(z0[1], z1[1], (frame%20)/20)
        line.set_data(ls, ls * g[0] + g[1])
        if frame%20==0: sleep(0.5)
        return line,
    def helper(xtb):
        try:
            X = float(xtb)
            Y = z[0][0]*X + z[0][1]
            xh.set_data([X, X], [min(y), Y])
            yh.set_data([min(x), X], [Y, Y])
            point.set_data([X], [Y])
            ax.set_title(f'Point at ({X:.2f}, {Y:.2f})')
            fig.canvas.draw()
        except ValueError: pass
    a = anim(fig, update, frames = 100, interval = 20, blit = True, repeat = False)
    box = TextBox(axes((0.2, 0.025, 0.6, 0.075)), 'Point', initial="10")
    helper(box.text)
    box.on_submit(helper)
    show()
