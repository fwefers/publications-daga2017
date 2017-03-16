#
#  Example script: Visualizes a trajectory
#  Author: Frank Wefers (fwefers@fwefers.de)
#

import numpy as np
from trajectory import *
np.set_printoptions(precision=6, suppress=True)

c = SampledTrajectory.load("data/curve1.traj")

plotTrajectory(c, markers=True)
figManager = plt.get_current_fig_manager()
#figManager.window.showMaximized()
plt.tight_layout()
plt.show()
