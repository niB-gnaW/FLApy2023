import FLApy as fp
import numpy as np
import pyvista as pv


def test_vis_3Dpoint():
    random_points = np.random.rand(100, 3)
    pc = fp.Visualization.vis_3Dpoint(random_points)
    pv.close_all()
    assert isinstance(pc, pv.plotting.qt_plotting.BackgroundPlotter) is True