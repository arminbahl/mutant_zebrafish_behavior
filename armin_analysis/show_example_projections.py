import numpy as np
import matplotlib
import my_figure as myfig
from pathlib import Path
import pylab as pl

root_path = Path("/Users/arminbahl/Dropbox/mutant_manuscript/paper/Review")

fig = myfig.Figure(title="Pareto fronts")

x_pos = 0
y_pos = 0

test_examples = [[[-5, 0], [3, 1], 90, 180, 1, 4],
                 [[-5, 0], [3, -3], 90, 180, 1, 2],
                 [[0, 5], [1, -1], 90, 45, 1, 4],
                 [[4, 5], [-2, -1], 70, 30, 1, 3],
                 [[4, 5], [6, -6], 70, 100, 1, 3],
                 ]

for pos_focal1, pos_other1, ang_focal, ang_other, speed_focal, speed_other in test_examples:

    pos_focal2 = np.array([pos_focal1[0] + np.cos(np.radians(ang_focal)) * speed_focal,
                           pos_focal1[1] + np.sin(np.radians(ang_focal)) * speed_focal])

    pos_other2 = np.array([pos_other1[0] + np.cos(np.radians(ang_other)) * speed_other,
                           pos_other1[1] + np.sin(np.radians(ang_other)) * speed_other])

    # Where on the eye, the point falls, datermines the weighing of the resulting motion vector
    ang2 = np.degrees(np.arctan2(pos_other1[1] - pos_focal1[1],
                                 pos_other1[0] - pos_focal1[0]))

    print(ang2, pos_focal1, pos_other1)

    if ang2 < 0:
        ang2 = 360 + ang2

    if ang2 > ang_focal:
        ang2 -= 360

    location_on_retina = ang_focal - ang2
    if location_on_retina > 180:
        location_on_retina = location_on_retina - 360

    # in the front, the weight is +1, in the back it is -1
    retina_weight = -np.abs(location_on_retina) / 90 + 1

    ang1 = np.degrees(np.arctan2(pos_other1[1] - pos_focal1[1],
                                 pos_other1[0] - pos_focal1[0]))
    ang2 = np.degrees(np.arctan2(pos_other2[1] - pos_focal1[1],
                                 pos_other2[0] - pos_focal1[0]))

    d_ang = ang2 - ang1
    d_ang = -((d_ang + 180 + 360) % 360 - 180)

    force = retina_weight*d_ang
    if force < 0:
        direction = "Left"
    else:
        direction = "Right"

    p0 = myfig.Plot(fig, num='', xpos=x_pos+4, ypos=y_pos + 10, plot_height=2.5, plot_width=2.5,
                    title=f"Motion arc: {d_ang:0.1f} deg;\nLocation: {location_on_retina:.1f} (Weight {retina_weight:.1f})\nMotion force: {force:.1f} ({direction})",
                    lw=1, pc='white', errorbar_area=False,
                    xl="X" if y_pos == 0 else None, xmin=-10, xmax=10, xticks=[-10, -5, 0, 5, 10] if y_pos == 0 else None,
                    yl="Y" if x_pos == 0 else None, ymin=-10, ymax=10, yticks=[-10, -5, 0, 5, 10] if x_pos == 0 else None,
                    hlines=[0], vlines=[0])

    p0.ax.arrow(pos_focal1[0], pos_focal1[1],
             pos_focal2[0] - pos_focal1[0], pos_focal2[1] - pos_focal1[1], width=0.2, color='C3')

    p0.ax.arrow(pos_other1[0], pos_other1[1],
             pos_other2[0] - pos_other1[0], pos_other2[1] - pos_other1[1], width=0.2, color='black')

    myfig.Line(p0, x=[pos_focal1[0], pos_other1[0]], y=[pos_focal1[1], pos_other1[1]], lc='C0')
    myfig.Line(p0, x=[pos_focal1[0], pos_other2[0]], y=[pos_focal1[1], pos_other2[1]], lc='C1')

    circle1 = pl.Circle(pos_focal1, 2, color='r', fill=False)
    p0.ax.add_patch(circle1)

    x_pos += 3.5
    if x_pos == 3.5*3:
        x_pos = 0
        y_pos += 3.5

# Show the retina weight function
p0 = myfig.Plot(fig, num='', xpos=4, ypos=23, plot_height=1.5, plot_width=1.5,
                    lw=1, pc='white', errorbar_area=False,
                    xl="Retina location", xmin=-190, xmax=190, xticks=[-180, -90, 0, 90, 180],
                    yl="Retina weight", ymin=-1.1, ymax=1.1, yticks=[-1, -0.5, 0, 0.5, 1],
                    hlines=[0], vlines=[0])

myfig.Line(p0, x=[-180, 0], y=[-1, 1], lc='black')
myfig.Line(p0, x=[180, 0], y=[-1, 1], lc='black')


fig.savepdf(root_path / f"example_fish_projections", open_pdf=True)