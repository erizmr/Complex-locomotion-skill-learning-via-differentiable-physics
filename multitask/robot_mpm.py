n_grid = 40
# n_grid = 64
dx = 1 / n_grid
squ = 0.05
n_squ = int(squ * n_grid)


class Scene:
    def __init__(self):
        self.n_particles = 0
        self.n_solid_particles = 0
        self.x = []
        self.actuator_id = []
        self.particle_type = []
        self.offset_x = 0
        self.offset_y = 0

    def add_rect(self, x, y, w, h, actuation, ptype=1):
        if w > squ or h > squ:
            dw = int(w / squ + 1e-6)
            dh = int(h / squ + 1e-6)
            for i in range(dw):
                for j in range(dh):
                    self.add_rect(x + i * squ, y + j * squ, squ, squ, actuation, ptype)
            return

        if ptype == 0:
            assert actuation == -1
        global n_particles
        w_count = int(w / dx) * 2
        h_count = int(h / dx) * 2
        real_dx = w / w_count
        real_dy = h / h_count
        for i in range(w_count):
            for j in range(h_count):
                self.x.append([
                    x + (i + 0.5) * real_dx + self.offset_x,
                    y + (j + 0.5) * real_dy + self.offset_y
                ])
                self.actuator_id.append(actuation)
                self.particle_type.append(ptype)
                self.n_particles += 1
                self.n_solid_particles += int(ptype == 1)

    def set_offset(self, x, y):
        self.offset_x = x
        self.offset_y = y

    def finalize(self):
        global n_particles, n_solid_particles
        n_particles = self.n_particles
        n_solid_particles = self.n_solid_particles
        print('n_particles', n_particles)
        print('n_solid', n_solid_particles)

    def set_n_actuators(self, n_act):
        global n_actuators
        n_actuators = n_act


def fish(scene):
    scene.add_rect(0.025, 0.025, 0.95, 0.1, -1, ptype=0)
    scene.add_rect(0.1, 0.2, 0.15, 0.05, -1)
    scene.add_rect(0.1, 0.15, 0.025, 0.05, 0)
    scene.add_rect(0.125, 0.15, 0.025, 0.05, 1)
    scene.add_rect(0.2, 0.15, 0.025, 0.05, 2)
    scene.add_rect(0.225, 0.15, 0.025, 0.05, 3)
    scene.set_n_actuators(4)


def robotA():
    scene = Scene()
    scene.set_offset(0.1, 0.1)
    scene.add_rect(0.0, 0.1, 0.3, 0.1, -1)
    scene.add_rect(0.0, 0.0, 0.05, 0.1, 0)
    scene.add_rect(0.05, 0.0, 0.05, 0.1, 1)
    scene.add_rect(0.2, 0.0, 0.05, 0.1, 2)
    scene.add_rect(0.25, 0.0, 0.05, 0.1, 3)
    scene.set_n_actuators(4)
    scene.finalize()
    return scene.x, scene.actuator_id, 4


robots_mpm = [robotA]