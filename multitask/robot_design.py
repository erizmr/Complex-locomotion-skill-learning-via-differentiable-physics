from util import read_json, write_json
from config_util import dump_to_json


class RobotDesignBase:
    def __init__(self, cfg):
        self.config = cfg

    def build(self):
        pass

    def clear(self):
        pass

    def update(self, file_name):
        self.config = read_json(file_name)
        self.clear()
        self.build()

    def draw(self):
        pass

    def get_objects(self):
        pass

    def dump_to_json(self, file):
        pass

    @classmethod
    def from_file(cls, file_name):
        config = read_json(file_name)
        return cls(config)


class RobotDesignMassSpring(RobotDesignBase):
    def __init__(self, cfg):
        super(RobotDesignMassSpring, self).__init__(cfg)
        self.robot_id = self.config["robot"]["id"]
        self.solver = self.config["robot"]["solver"]
        # design data holders
        self.objects = []
        self.springs = []

        self.points = []
        self.point_id = []
        self.mesh_springs = []

        self.built = False

    def build(self):
        stiffnesses = self.config["design"]["physical_parameter"]["stiffness"]
        actuations = self.config["design"]["physical_parameter"]["actuation"]
        mesh_types = self.config["design"]["mesh_type"]
        actuation_enables = self.config["design"]["actuation_enable"]
        active_spring_mechanisms = self.config["design"]["active_spring_mechanism"]

        for i, anchor in enumerate(self.config["design"]["anchor"]):
            x, y = anchor[0], anchor[1]
            mesh_type = mesh_types[i]
            actuation = actuations[i]
            stiffness = stiffnesses[i]
            actuation_enable = actuation_enables[i]
            active_spring_mechanism = active_spring_mechanisms[i]

            if actuation_enable == 1:
                active_spring = self.config["active_spring_template"][mesh_type][active_spring_mechanism]
            else:
                active_spring = self.config["active_spring_template"][mesh_type]["dummy"]

            if mesh_type == "square":
                self.add_mesh_square(x, y,
                                     stiffness=stiffness,
                                     actuation=actuation,
                                     active_spring=active_spring)
            elif mesh_type == "triangle":
                self.add_mesh_triangle(x, y,
                                       stiffness=stiffness,
                                       actuation=actuation,
                                       active_spring=active_spring)
            # elif mesh_type == "square_big":
            #     self.add_mesh_square_big(x, y,
            #                              stiffness=stiffness,
            #                              actuation=actuation,
            #                              active_spring=active_spring)
            else:
                raise NotImplementedError(f"{mesh_type} not implemented.")
        self.built = True

        return self.get_objects()

    def clear(self):
        self.objects.clear()
        self.springs.clear()
        self.points.clear()
        self.point_id.clear()
        self.mesh_springs.clear()
        self.built = False

    def get_objects(self):
        assert self.built
        return self.objects, self.springs

    def dump_to_json(self, file=None):
        assert self.built
        return dump_to_json(solver=self.solver,
                            objects=self.objects,
                            springs=self.springs,
                            file=file)

    def draw(self):
        assert self.built
        import taichi as ti
        gui = ti.GUI(background_color=0xFFFFFF)

        def circle(x, y, color):
            gui.circle((x, y), ti.rgb_to_hex(color), 7)
        while gui.running:
            # draw segments
            for i in range(len(self.springs)):
                def get_pt(x):
                    return x[0], x[1]
                r = 2
                c = 0x222222
                # Show active spring in red
                a = self.springs[i][4] * 1.8
                if a > 0.:
                    r = 4
                    c = ti.rgb_to_hex((0.5 + a, 0.5 - abs(a), 0.5 - a))
                gui.line(get_pt(self.objects[self.springs[i][0]]),
                         get_pt(self.objects[self.springs[i][1]]),
                         color=c,
                         radius=r)
            # draw points
            for i in range(len(self.objects)):
                color = (0.06640625, 0.06640625, 0.06640625)
                circle(self.objects[i][0], self.objects[i][1], color)
            gui.show()

    def add_object(self, x):
        self.objects.append(x)
        return len(self.objects) - 1

    def add_spring(self, a, b, length=None, stiffness=1, actuation=0.1):
        if length is None:
            length = ((self.objects[a][0] - self.objects[b][0]) ** 2 +
                      (self.objects[a][1] - self.objects[b][1]) ** 2) ** 0.5
        self.springs.append([a, b, length, stiffness, actuation])

    def add_mesh_point(self, i, j):
        if (i, j) not in self.points:
            id = self.add_object((i * 0.05 + 0.1, j * 0.05 + 0.1))
            self.points.append((i, j))
            self.point_id.append(id)
        return self.point_id[self.points.index((i, j))]

    def add_mesh_spring(self, a, b, s, act):
        if (a, b) in self.mesh_springs or (b, a) in self.mesh_springs:
            return

        self.mesh_springs.append((a, b))
        self.add_spring(a, b, stiffness=s, actuation=act)

    def add_mesh_square(self, i, j, stiffness=3e4, actuation=0.0, active_spring=None):
        a = self.add_mesh_point(i, j)
        b = self.add_mesh_point(i, j + 1)
        c = self.add_mesh_point(i + 1, j)
        d = self.add_mesh_point(i + 1, j + 1)

        if active_spring is None:
            active_spring = [1]*6

        # b d
        # a c
        link_order = [[a, b], [a, c], [a, d], [b, c], [b, d], [c, d]]
        for i, link in enumerate(link_order):
            self.add_mesh_spring(link[0], link[1], stiffness, actuation * active_spring[i])

        # self.add_mesh_spring(a, b, stiffness, actuation)
        # self.add_mesh_spring(a, c, stiffness, 0)
        # self.add_mesh_spring(a, d, stiffness, 0)
        # self.add_mesh_spring(b, c, stiffness, 0)
        # self.add_mesh_spring(b, d, stiffness, 0)
        # self.add_mesh_spring(c, d, stiffness, actuation)

    def add_mesh_triangle(self, i, j, stiffness=3e4, actuation=0.0, active_spring=None):
        a = self.add_mesh_point(i + 0.5, j + 0.5)
        b = self.add_mesh_point(i, j + 1)
        d = self.add_mesh_point(i + 1, j + 1)

        if active_spring is None:
            active_spring = [1]*3

        # b     d
        #    a
        #
        link_order = [[a, b], [a, d], [b, d]]
        for i, link in enumerate(link_order):
            self.add_mesh_spring(i, j, stiffness, actuation * active_spring[i])
        # for i in [a, b, d]:
        #     for j in [a, b, d]:
        #         if i != j:
        #             self.add_mesh_spring(i, j, 3e4, 0)

    # def add_mesh_square_big(self, x, y, stiffness=3e4, actuation=0.0, active_spring=None):
    #     self.add_mesh_square(x * 2, y * 2,
    #                          stiffness=stiffness,
    #                          actuation=actuation,
    #                          active_spring=active_spring)
    #     self.add_mesh_square(x * 2, y * 2 + 1,
    #                          stiffness=stiffness,
    #                          actuation=actuation,
    #                          active_spring=active_spring)
    #     self.add_mesh_square(x * 2 + 1, y * 2,
    #                          stiffness=stiffness,
    #                          actuation=actuation,
    #                          active_spring=active_spring)
    #     self.add_mesh_square(x * 2 + 1, y * 2 + 1,
    #                          stiffness=stiffness,
    #                          actuation=actuation,
    #                          active_spring=active_spring)
