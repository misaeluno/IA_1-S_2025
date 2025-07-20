from matplotlib import cm
from matplotlib.pyplot import subplots, axes, show, pause, figure, ion, ioff
from matplotlib.widgets import TextBox
from mpl_toolkits.mplot3d import Axes3D
from numpy import fix, linspace, ones, sum as Σ, column_stack, loadtxt, array, meshgrid
from sys import argv

def helper(xtb):
    try:
        Y = θ[1, 0]*(X := float(xtb)) + θ[0, 0]
        xh.set_data([X, X], [y[:, 0].min(), Y])
        yh.set_data([x[:, 1].min(), X], [Y, Y])
        π1.set_data([X], [Y])
        ax1.set_title(f'Regresión Lineal, α = {α}, punto en ({X:.2f}, {Y:.2f})')
    except ValueError: pass

if len(argv) < 5: print('Проблеми с аргументом.'); exit()

α, n, и, ѳ, й = float(argv[1]), int(argv[2]), [(θ := array(eval(argv[3]), dtype=float).reshape(-1, 1)).flatten()], float(argv[4]), []
x, y = column_stack([ones(len(д := loadtxt(open('C:/Users/DEATH/Downloads/data1.txt', 'r'), delimiter=","))), д[:, 0]]), д[:, 1].reshape(-1, 1)
hᶿ = lambda: x @ θ
j = lambda: Σ((hᶿ() - y)**2) / (2 * x.shape[0])
Δ = lambda: (x.T @ (hᶿ() - y)) / x.shape[0]

ls = linspace(x[:, 1].min(), x[:, 1].max(), 100)

ion()
ф1 = figure(figsize = (12, 6))
й.append(float(j()))

ax1 = ф1.add_subplot(); ax1.set_position((0.1, 0.25, 0.35, 0.65)); ax1.set_xlabel('Población de la ciudad en 10\'000s'); ax1.set_ylabel('Beneficio en $10\'000s')
ax1.grid(True, linestyle = '--', alpha = .4); ax1.set_xlim(x[:, 1].min(), x[:, 1].max()); ax1.set_ylim(y[:, 0].min(), y[:, 0].max())
ax1.scatter(x[:, 1], y[:, 0], c = 'blue', marker = 'x', alpha = 0.6, s = 50, label = 'Datos de entrenamiento')
l1, = ax1.plot(x[:, 1], hᶿ(), 'r-', label = 'Regresión Lineal'); xh, = ax1.plot([], [], 'r--'); yh, = ax1.plot([], [], 'r--'); π1, = ax1.plot([], [], 'ro')
љ = ax1.legend(loc = 'lower right', framealpha = 0.8, edgecolor = 'black', facecolor = 'white', bbox_to_anchor = (0.95, 0.05), fancybox = True, frameon = True)
љ.get_frame().set_boxstyle("Round", pad = 0.3)
љ.set_draggable(True)
box = TextBox(ф1.add_axes((0.2, 0.025, 0.6, 0.075)), 'x', initial="20")
helper(20)

ax2 = ф1.add_subplot(); ax2.set_position((0.55, 0.25, 0.35, 0.65)); ax2.set_title(f'Historial de la Función de Costo'); ax2.set_xlabel('Iteración'); ax2.set_ylabel('Costo')
ax2.grid(True, linestyle = '--', alpha = .4); ax2.set_xlim(0, n); ax2.set_ylim(4, й[0])
l2, = ax2.plot([], [], 'r-')
θₜ = ax2.text(0.95, 0.95, f'Nᵢ = 0\nθ₀ = {θ[0,0]:.4f}\nθ₁ = {θ[1,0]:.4f}\nΔй = ∞', transform = ax2.transAxes, ha = 'right', va = 'top',
    bbox=dict(facecolor = 'white', alpha = 0.8, edgecolor = 'black', boxstyle = 'round'))

ф2 = figure(figsize = (12, 6))
ax3 = ф2.add_subplot(); ax3.set_position((0.1, 0.1, 0.35, 0.8)); ax3.set_title('Trazas del Descenso del Gradiente'); ax3.set_xlabel(f'θ₀ = {θ[0,0]:.4f}'); ax3.set_ylabel(f'θ₁ = {θ[1,0]:.4f}')
ax3.grid(True, linestyle = '--', alpha = .4); ax3.set_xlim(θ[0,0] - 1, θ[0,0] + 1); ax3.set_ylim(θ[1,0] - 1, θ[1,0] + 1)
l3, = ax3.plot([], [], 'r-')
θ0м, θ1м = meshgrid(linspace(θ[0,0] - 1, θ[0,0] + 1, 10), linspace(θ[1,0] - 1, θ[1,0] + 1, 10))
ψ = array([Σ((x @ array([θ0, θ1]).reshape(-1,1) - y)**2) / (2 * x.shape[0]) for θ0, θ1 in zip(θ0м.ravel(), θ1м.ravel())]).reshape(θ0м.shape)
к1 = ax3.contour(θ0м, θ1м, ψ, levels = linspace(0, 10, 20), cmap = 'coolwarm')
þ1, = ax3.plot([], [], 'r-', lw=2, label='Descenso del Gradiente'); π2, = ax3.plot([], [], 'ro', markersize=8)

ax4 = ф2.add_subplot(projection = '3d'); ax4.set_position((0.55, 0.1, 0.35, 0.8)); ax4.set_title('Descenso del Gradiente 😈')
ax4.set_xlim(θ[0,0] - 1, θ[0,0] + 1); ax4.set_ylim(θ[1,0] - 1, θ[1,0] + 1); ax4.set_xlabel('θ₀'); ax4.set_ylabel('θ₁'); ax4.set_zlabel(f'Costo J(θ) = {й[0]}')
σ1 = ax4.plot_surface(θ0м, θ1м, ψ, cmap = 'coolwarm', alpha = 0.7, rstride = 1, cstride = 1)
þ2, = ax4.plot([], [], [], 'r-', lw=2, label='Trayectoria')
π3, = ax4.plot([], [], [], 'ro', markersize=8)

pause(3)
for _ in range(abs(n + 1)):
    for c in ax3.collections: c.remove()
    # ax3
    ax3.set_xlim(θ[0,0] - 1, θ[0,0] + 1); ax3.set_ylim(θ[1,0] - 1, θ[1,0] + 1)
    θ0м, θ1м = meshgrid(linspace(θ[0,0] - 1, θ[0,0] + 1, 10), linspace(θ[1,0] - 1, θ[1,0] + 1, 10))
    ψ = array([Σ((x @ array([θ0, θ1]).reshape(-1,1) - y)**2) / (2 * x.shape[0]) for θ0, θ1 in zip(θ0м.ravel(), θ1м.ravel())]).reshape(θ0м.shape)
    к1 = ax3.contour(θ0м, θ1м, ψ, levels = linspace(0, 100, 20), cmap = 'coolwarm')
    þ1.set_data([т[0] for т in и], [т[1] for т in и])
    π2.set_data([θ[0,0]], [θ[1,0]])
    # ax4
    for a in ax4.collections + ax4.lines + ax4.texts:
        a.remove()
    σ1 = ax4.plot_surface(θ0м, θ1м, ψ, cmap='coolwarm', alpha=0.7, rstride=1, cstride=1)
    ax4.contour(θ0м, θ1м, ψ, zdir='z', offset=ψ.min()-5, cmap='coolwarm', alpha=0.5)  # Offset below surface
    þ2, = ax4.plot([], [], [], 'r-', lw=2)  # Recreate path line
    π3, = ax4.plot([], [], [], 'ro', markersize=8)  # Recreate point
    þ2.set_data_3d(array([т[0] for т in и]).flatten(), array([т[1] for т in и]).flatten(), array(й).flatten())
    π3.set_data_3d([θ[0,0]], [θ[1,0]], [й[-1]])
    μ = (Σ(й) / (_ + 1)) + 4
    ax4.set_xlim(θ[0,0] - 1, θ[0,0] + 1); ax4.set_ylim(θ[1,0] - 1, θ[1,0] + 1); ax4.set_zlim(4, μ); ax4.set_zlabel(f'Costo J(θ) = {й[-1]}')
    # ax1
    и.append((θ := θ - α * Δ()).flatten())
    й.append(float(j()))
    l1.set_ydata(hᶿ())
    helper(20)
    # ax2
    l2.set_data(range(len(й)), й)
    ax2.set_ylim(4, μ)
    θₜ.set_color('red' if й[-1] > й[-2] else 'black')
    θₜ.set_text(f'Nᵢ = {_}\nθ₀ = {θ[0,0]:.4f}\nθ₁ = {θ[1,0]:.4f}\nΔй = {abs(й[-1] - й[-2]):.4f}'
        if abs(й[-1] - й[-2]) > ѳ else
        f'Nᵢ = {_}😳\nθ₀ = {θ[0,0]:.4f} 😈\nθ₁ = {θ[1,0]:.4f} ❤\nΔй = {abs(й[-1] - й[-2]):.4f}😩')
    if abs(й[-1] - й[-2]) < ѳ:
        break
    ax3.set_xlabel(f'θ₀ = {θ[0,0]:.4f}'); ax3.set_ylabel(f'θ₁ = {θ[1,0]:.4f}')
    ф1.canvas.draw_idle()
    ф2.canvas.draw_idle()

    pause(0.05)
#print(f'Final θ:\n{θ}\nHistory:\n{и}')
box.on_submit(helper)
ioff()
show()
