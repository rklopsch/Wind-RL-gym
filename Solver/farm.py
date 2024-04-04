import matplotlib.pyplot as plt
import matplotlib.patches as patch
import numpy as np
import math
import copy
import random
# import fonts


class Farm:
    def __init__(self, lx, lz, n, base_turbine, offset=None):
        if offset is None:
            offset = [0., 0.]
        self.lx = lx
        self.lz = lz
        self.offset = offset
        self.base_turbine = base_turbine
        self.n_turbines = n
        self.turbines = [None] * n
        for i in range(self.n_turbines):
            self.turbines[i] = copy.deepcopy(base_turbine)
        self.minx = self.offset[0]
        self.maxx = self.offset[0]+self.lx
        self.minz = self.offset[1]
        self.maxz = self.offset[1]+self.lz

    def plot_turbines(self, ax):
        for turbine in self.turbines:
            turbine.plot_turbine(ax)

    def draw_bounds(self, ax):
        ax.add_patch(patch.Rectangle(self.offset, self.lx, self.lz,
                                     facecolor="none", ec='r', lw=2, linestyle='--'))
        ax.add_patch(patch.Rectangle([x-self.turbines[0].diam/2 for x in self.offset],
                                     self.lx + self.turbines[0].diam,
                                     self.lz + self.turbines[0].diam,
                                     facecolor="none", ec='r', lw=2, linestyle=':'))

    def grid(self, staggered=False):
        spacing = math.sqrt(self.lx*self.lz/self.n_turbines)
        # Fill lz first then calculate nx
        nz = math.ceil(self.lz/spacing)
        nx = math.ceil(self.n_turbines/nz)
        # adjust spacings to fill box
        dx = self.lx/max(nx-1, 1)
        dz = self.lz/max(nz-1+staggered, 1)
        for i in range(self.n_turbines):
            turbine = self.turbines[i]
            turbine.location[0] = self.offset[0] + math.floor(i/nz) * dx
            turbine.location[1] = self.offset[1] + i % nz * dz + staggered*((i/nz) % 2 * dz/2)

    def scatter(self):
        for i in range(self.n_turbines):
            turbine = self.turbines[i]
            turbine.location[0] = self.offset[0] + random.uniform(0, self.lx)
            turbine.location[1] = self.offset[1] + random.uniform(0, self.lz)

    def write_adm(self, filename='adm'):
        # Writing to file
        with open(f"{filename}.ad", "w") as file:
            # Writing data to a file
            file.write("!CoR(x) CoR(y) CoR(z) YawAng[deg] TiltAng[deg] RotorDiam C_T[-] alpha[-] \n")
            for turbine in self.turbines:
                file.write(f"{turbine.location[0]+self.offset[0]} "
                           f"{turbine.hub_height} "
                           f"{turbine.location[1]+self.offset[1]} "
                           f"{turbine.yaw} "
                           f"{turbine.tilt} "
                           f"{turbine.diam} "
                           f"{turbine.c_t} "
                           f"{turbine.alpha} \n")

    def get_layout(self):
        x = np.zeros(self.n_turbines)
        y = np.zeros(self.n_turbines)
        for i in range(self.n_turbines):
            turbine = self.turbines[i]
            x[i] = turbine.location[0]
            y[i] = turbine.location[1]
        return x, y

    def set_yaw(self, yaws):
        if len(yaws) != self.n_turbines:
            print(f"Number of yaws ({len(yaws)}) should equal number of turbines ({self.n_turbines})")
        else:
            for i in range(self.n_turbines):
                turbine = self.turbines[i]
                turbine.yaw = yaws.squeeze()[i]


class Turbine:
    def __init__(self, diam, hub_height, location=None, yaw=0., tilt=0.0, c_t=0.75, alpha=0.17095):
        if location is None:
            location = [0, 0]
        self.diam = diam
        self.hub_height = hub_height
        self.location = location
        self.yaw = yaw
        self.tilt = tilt
        self.c_t = c_t        # Thrust coefficent
        self.alpha = alpha   # Induction Coefficient

    def plot_turbine(self, ax):
        x = [self.location[0]-self.diam/2*math.sin(self.yaw),
             self.location[0]+self.diam/2*math.sin(self.yaw)]
        z = [self.location[1]-self.diam/2*math.cos(self.yaw),
             self.location[1]+self.diam/2*math.cos(self.yaw)]
        ax.plot(x, z, c='k')
        ax.scatter(self.location[0], self.location[1], c='k')


def main():
    farm1 = Farm(1500, 1001, 25, Turbine(100, 100, yaw=0), offset=[800, 770])
    farm1 = Farm(126 * 14, 126*4, 3, Turbine(126, 90, yaw=0), offset=[2 * 126, 2*126])

    farm1.grid(staggered=False)
    # farm1.scatter()

    f, a = plt.subplots(1, 1, figsize=(10, 10), constrained_layout=True)
    a.set_aspect('equal')
    farm1.plot_turbines(a)
    farm1.draw_bounds(a)
    farm1.write_adm('test')
    a.set_xlabel(r'$x$')
    a.set_ylabel(r'$z$')

    plt.show()


if __name__ == "__main__":
    main()


