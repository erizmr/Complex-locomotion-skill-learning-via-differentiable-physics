from util import read_json, write_json
from multitask.config_util import dump_to_json

import numpy as np
import json


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


class RobotDesignMPM(RobotDesignBase):
    def __init__(self, cfg):
        super(RobotDesignMPM, self).__init__(cfg)
        assert self.config["robot"]["solver"] == "mpm"
        self.robot_id = self.config["robot"]["id"]
        self.robot_name = self.config["robot"]["name"]
        # design data holders
        self.n_particles = 0
        self.n_solid_particles = 0
        self.pos = []
        self.actuator_id = []

        self.square_size = self.config["design_parameter"]["square_size"]

        self.n_num = self.config["design_parameter"]["particle_in_square_edge"]

        self.actuator_num = self.config["design_parameter"]["actuator_num"]

        if "offset" in self.config["design_parameter"].keys():
            self.offset = np.array((
                self.config["design_parameter"]["offset"]["x"],
                self.config["design_parameter"]["offset"]["y"]))
        else:
            self.offset = np.array([0, 0])

        self.built = False

    def add_particle(self, pos, act_id):
        assert act_id >= -1 and act_id < self.actuator_num
        self.pos.append(self.offset + pos)
        self.actuator_id.append(act_id)
        self.n_particles += 1
        self.n_solid_particles += int(act_id == -1)

    def add_square(self, pos, act_id):
        dx = self.square_size / self.n_num
        for i in range(self.n_num):
            for j in range(self.n_num):
                self.add_particle(
                    (pos[0] * self.square_size + (i + 0.5) * dx,
                     pos[1] * self.square_size + (j + 0.5) * dx),
                    act_id)

    def add_rect(self, pos, shape, act_id):
        for i in range(shape[0]):
            for j in range(shape[1]):
                self.add_square(
                    (pos[0] + i, pos[1] + j), act_id)

    def get_objects(self):
        assert self.built
        return self.pos, self.actuator_id, self.actuator_num

    def build(self):
        positions = self.config["design"]["anchor"]
        shapes = self.config["design"]["shape"]
        act_ids = self.config["design"]["actuator"]
        mesh_types = self.config["design"]["mesh_type"]
        for id, mesh_type in enumerate(mesh_types):
            if mesh_type == "rectangle":
                self.add_rect(positions[id], shapes[id], act_ids[id])
            else:
                raise NotImplementedError(
                    "{} mesh not implemented!".format(mesh_type))
        self.built = True
        return self.get_objects()

    def clear(self):
        self.pos.clear()
        self.actuator_id.clear()
        self.n_particles = 0
        self.n_solid_particles = 0
        self.built = False


class RobotDesignMassSpring(RobotDesignBase):
    def __init__(self, cfg):
        super(RobotDesignMassSpring, self).__init__(cfg)
        assert self.config["robot"]["solver"] == "mass_spring"
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
        anchors = self.config["design"]["anchor"]
        elements_num = len(anchors)
        mesh_types = self.config["design"]["mesh_type"]
        assert len(mesh_types) == elements_num
        stiffnesses = self.config["design"]["physical_parameter"]["stiffness"]
        assert len(stiffnesses) == elements_num
        actuations = self.config["design"]["physical_parameter"]["actuation"]
        assert len(actuations) == elements_num
        actuation_enables = self.config["design"]["actuation_enable"]
        assert len(actuation_enables) == elements_num
        active_spring_mechanisms = self.config["design"]["active_spring_mechanism"]
        assert len(active_spring_mechanisms) == elements_num

        for i, anchor in enumerate(anchors):
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
        objects_for_dump = []
        springs_for_dump = []
        for obj in self.objects:
            objects_for_dump.append({"x": obj[0], "y": obj[1]})
        for spr in self.springs:
            springs_for_dump.append({"vertex_1": spr[0],
                                     "vertex_2": spr[1],
                                     "length": spr[2],
                                     "stiffness": spr[3],
                                     "actuation": spr[4]})

        return dump_to_json(solver=self.solver,
                            objects=objects_for_dump,
                            springs=springs_for_dump,
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

class RobotDesignMassSpring3D(RobotDesignBase):
    def __init__(self, cfg):
        super(RobotDesignMassSpring3D, self).__init__(cfg)
        assert self.config["robot"]["solver"] == "mass_spring"
        self.robot_id = self.config["robot"]["id"]
        self.solver = self.config["robot"]["solver"]
        self.spring_stiffness = self.config["robot"]["spring_stiffness"]
        self.spring_actuation = self.config["robot"]["spring_actuation"]
        print("Spring stiffness ", self.spring_stiffness)
        print("Spring actuation ", self.spring_actuation)
        # design data holders
        self.objects = []
        self.springs = []

        self.points = []
        self.point_id = []
        self.mesh_springs = []
        self.faces = []

        self.built = False

    def add_object(self, x):
        self.objects.append(x)
        return len(self.objects) - 1

    def add_spring(self, a, b, length=None, stiffness=1, actuation=0.1):
        if length == None:
            length = ((self.objects[a][0] - self.objects[b][0])**2 +
                      (self.objects[a][1] - self.objects[b][1])**2 +
                      (self.objects[a][2] - self.objects[b][2])**2)**0.5
        self.springs.append([a, b, length, stiffness, actuation])

    def add_mesh_point(self, i, j, k):
        if (i, j, k) not in self.points:
            id = self.add_object((i * 0.05 + 0.1, j * 0.05 + 0.1, k * 0.05 + 0.1))
            self.points.append((i, j, k))
            self.point_id.append(id)
        return self.point_id[self.points.index((i, j, k))]

    def add_foot_point(self, i, j, k):
        if (i, j, k) not in self.points:
            id = self.add_object((i * 0.05 + 0.125, j * 0.05 + 0.1, k * 0.05 + 0.125))
            self.points.append((i, j, k))
            self.point_id.append(id)
        return self.point_id[self.points.index((i, j, k))]

    def add_mesh_spring(self, a, b, s, act):
        if (a, b) in self.mesh_springs or (b, a) in self.mesh_springs:
            return

        self.mesh_springs.append((a, b))
        self.add_spring(a, b, stiffness=s, actuation=act)

    def add_mesh_square(self, i, j, k, actuation=0.0):
        a = self.add_mesh_point(i, j, k)
        b = self.add_mesh_point(i, j + 1, k)
        c = self.add_mesh_point(i + 1, j, k)
        d = self.add_mesh_point(i + 1, j + 1, k)
        e = self.add_mesh_point(i, j, k + 1)
        f = self.add_mesh_point(i, j + 1, k + 1)
        g = self.add_mesh_point(i + 1, j, k + 1)
        h = self.add_mesh_point(i + 1, j + 1, k + 1)

        # b d
        # a c
        self.add_mesh_spring(a, b, self.spring_stiffness, actuation)
        self.add_mesh_spring(c, d, self.spring_stiffness, actuation)
        self.add_mesh_spring(e, f, self.spring_stiffness, actuation)
        self.add_mesh_spring(g, h, self.spring_stiffness, actuation)

        self.add_mesh_spring(b, d, self.spring_stiffness, 0)
        self.add_mesh_spring(a, c, self.spring_stiffness, 0)
        self.add_mesh_spring(f, h, self.spring_stiffness, 0)
        self.add_mesh_spring(e, g, self.spring_stiffness, 0)

        self.add_mesh_spring(b, f, self.spring_stiffness, 0)
        self.add_mesh_spring(d, h, self.spring_stiffness, 0)
        self.add_mesh_spring(a, e, self.spring_stiffness, 0)
        self.add_mesh_spring(c, g, self.spring_stiffness, 0)

        self.add_mesh_spring(b, g, self.spring_stiffness, 0)
        self.add_mesh_spring(d, e, self.spring_stiffness, 0)
        self.add_mesh_spring(f, c, self.spring_stiffness, 0)
        self.add_mesh_spring(h, a, self.spring_stiffness, 0)

        self.add_mesh_spring(e, c, self.spring_stiffness, 0)
        self.add_mesh_spring(a, g, self.spring_stiffness, 0)
        self.add_mesh_spring(h, b, self.spring_stiffness, 0)
        self.add_mesh_spring(d, f, self.spring_stiffness, 0)

        self.add_mesh_spring(e, b, self.spring_stiffness, actuation)
        self.add_mesh_spring(a, f, self.spring_stiffness, actuation)
        self.add_mesh_spring(h, c, self.spring_stiffness, actuation)
        self.add_mesh_spring(d, g, self.spring_stiffness, actuation)

        self.add_mesh_spring(f, g, self.spring_stiffness, actuation)
        self.add_mesh_spring(e, h, self.spring_stiffness, actuation)
        self.add_mesh_spring(a, d, self.spring_stiffness, actuation)
        self.add_mesh_spring(c, b, self.spring_stiffness, actuation)

        def append_square_face(a, b, c, d):
            self.faces.append((a, b, c))
            self.faces.append((d, a, c))
        append_square_face(a, b, d, c)
        append_square_face(b, f, h, d)
        append_square_face(f, e, g, h)
        append_square_face(e, a, c, g)
        append_square_face(h, g, c, d)
        append_square_face(b, a, e, f)

    def robotA(self):
        self.add_mesh_square(0, 0, 0, actuation=self.spring_actuation)
        self.add_mesh_square(2, 0, 2, actuation=self.spring_actuation)
        self.add_mesh_square(0, 0, 2, actuation=self.spring_actuation)
        self.add_mesh_square(2, 0, 0, actuation=self.spring_actuation)

        self.add_mesh_square(0, 1, 0, actuation=0)
        self.add_mesh_square(0, 1, 1, actuation=0)
        self.add_mesh_square(0, 1, 2, actuation=0)
        self.add_mesh_square(1, 1, 0, actuation=0)
        self.add_mesh_square(1, 1, 1, actuation=0)
        self.add_mesh_square(1, 1, 2, actuation=0)
        self.add_mesh_square(2, 1, 0, actuation=0)
        self.add_mesh_square(2, 1, 1, actuation=0)
        self.add_mesh_square(2, 1, 2, actuation=0)

    def robotB(self):
        self.add_mesh_square(0, 0, 1, actuation=self.spring_actuation)
        self.add_mesh_square(1, 0, 0, actuation=self.spring_actuation)
        self.add_mesh_square(1, 0, 2, actuation=self.spring_actuation)
        self.add_mesh_square(2, 0, 1, actuation=self.spring_actuation)

        self.add_mesh_square(0, 1, 1, actuation=0)
        self.add_mesh_square(1, 1, 0, actuation=0)
        self.add_mesh_square(1, 1, 2, actuation=0)
        self.add_mesh_square(2, 1, 1, actuation=0)
        self.add_mesh_square(1, 1, 1, actuation=0)
        self.add_mesh_square(1, 2, 1, actuation=0)

    def robotC(self):
        self.add_mesh_square(0, 0, 0, actuation=self.spring_actuation)
        self.add_mesh_square(2, 0, 0, actuation=self.spring_actuation)
        self.add_mesh_square(1, 0, 2, actuation=self.spring_actuation)

        self.add_mesh_square(0, 1, 0, actuation=0)
        self.add_mesh_square(2, 1, 0, actuation=0)
        self.add_mesh_square(1, 1, 2, actuation=0)
        self.add_mesh_square(0, 1, 1, actuation=0)
        self.add_mesh_square(1, 1, 1, actuation=0)
        self.add_mesh_square(2, 1, 1, actuation=0)

    def robotD(self):
        with open('cfg3d/skeleton.json') as json_file:
            data = json.load(json_file)
        for v in data['nodes']:
            pos = data['nodes'][v]
            pos[0] = pos[0] * 0.05
            pos[1] = pos[1] * 0.05 + 0.212
            pos[2] = pos[2] * 0.05 * 0.75
            self.objects.append(pos)

        motors = {(9, 21), (36, 24), (29, 26), (14, 11),
            (14, 26), (10, 9), (29, 11), (25, 24), (25, 26), (36, 9), (21, 24), (10, 11)}

        s = set()
        for e in data['links']:
            a, b = e
            s.add((a, b))
            s.add((b, a))
            if (a, b) in motors or (b, a) in motors:
                self.add_mesh_spring(a, b, self.spring_stiffness, self.spring_actuation)
            else:
                self.add_mesh_spring(a, b, self.spring_stiffness, 0.)

        for e in list(motors) + [(14, 29), (29, 36), (36, 21), (21, 14), (11, 5), (26, 5), (24, 35), (9, 20), (26, 35), (11, 20), (9, 8), (24, 8), (11, 31), (26, 16), (24, 23), (9, 38)]:
            a, b = e
            if (a, b) in motors or (b, a) in motors:
                self.add_mesh_spring(a, b, self.spring_stiffness, self.spring_actuation)
            else:
                self.add_mesh_spring(a, b, self.spring_stiffness, 0.)

        for a in range(len(self.objects)):
            for b in range(len(self.objects)):
                for c in range(len(self.objects)):
                    if a < b < c and (a, b) in s and (b, c) in s and (a, c) in s:
                        self.faces.append((a, b, c))
        return self.objects, self.springs, self.faces

    def add_mesh_lying(self, i, j, k, actuation=0.0):
        a = self.add_mesh_point(i, j, k)
        b = self.add_mesh_point(i, j + 1, k)
        c = self.add_mesh_point(i + 1, j, k)
        d = self.add_mesh_point(i + 1, j + 1, k)
        e = self.add_mesh_point(i, j, k + 1)
        f = self.add_mesh_point(i, j + 1, k + 1)
        g = self.add_mesh_point(i + 1, j, k + 1)
        h = self.add_mesh_point(i + 1, j + 1, k + 1)

        # b d
        # a c
        self.add_mesh_spring(a, b, self.spring_stiffness, 0)
        self.add_mesh_spring(c, d, self.spring_stiffness, 0)
        self.add_mesh_spring(e, f, self.spring_stiffness, 0)
        self.add_mesh_spring(g, h, self.spring_stiffness, 0)

        self.add_mesh_spring(b, d, self.spring_stiffness, 0)
        self.add_mesh_spring(a, c, self.spring_stiffness, 0)
        self.add_mesh_spring(f, h, self.spring_stiffness, 0)
        self.add_mesh_spring(e, g, self.spring_stiffness, 0)

        self.add_mesh_spring(b, f, self.spring_stiffness, actuation)
        self.add_mesh_spring(d, h, self.spring_stiffness, actuation)
        self.add_mesh_spring(a, e, self.spring_stiffness, actuation)
        self.add_mesh_spring(c, g, self.spring_stiffness, actuation)

        self.add_mesh_spring(b, g, self.spring_stiffness, actuation)
        self.add_mesh_spring(d, e, self.spring_stiffness, actuation)
        self.add_mesh_spring(f, c, self.spring_stiffness, actuation)
        self.add_mesh_spring(h, a, self.spring_stiffness, actuation)

        self.add_mesh_spring(e, c, self.spring_stiffness, actuation)
        self.add_mesh_spring(a, g, self.spring_stiffness, actuation)
        self.add_mesh_spring(h, b, self.spring_stiffness, actuation)
        self.add_mesh_spring(d, f, self.spring_stiffness, actuation)

        self.add_mesh_spring(e, b, self.spring_stiffness, actuation)
        self.add_mesh_spring(a, f, self.spring_stiffness, actuation)
        self.add_mesh_spring(h, c, self.spring_stiffness, actuation)
        self.add_mesh_spring(d, g, self.spring_stiffness, actuation)

        self.add_mesh_spring(f, g, self.spring_stiffness, 0)
        self.add_mesh_spring(e, h, self.spring_stiffness, 0)
        self.add_mesh_spring(a, d, self.spring_stiffness, 0)
        self.add_mesh_spring(c, b, self.spring_stiffness, 0)

        def append_square_face(a, b, c, d):
            self.faces.append((a, b, c))
            self.faces.append((d, a, c))
        append_square_face(a, b, d, c)
        append_square_face(b, f, h, d)
        append_square_face(f, e, g, h)
        append_square_face(e, a, c, g)
        append_square_face(h, g, c, d)
        append_square_face(b, a, e, f)

    def robotE(self):
        self.add_mesh_lying(0, 0, 0, actuation=self.spring_actuation)
        self.add_mesh_lying(0, 0, 1, actuation=self.spring_actuation)
        self.add_mesh_lying(0, 0, 2, actuation=self.spring_actuation)
        self.add_mesh_lying(0, 0, 3, actuation=self.spring_actuation)
        self.add_mesh_lying(0, 0, 4, actuation=self.spring_actuation)
        self.add_mesh_lying(0, 0, 5, actuation=self.spring_actuation)

    def build(self):
        if self.robot_id == 100:
            self.robotA()
        elif self.robot_id == 101:
            self.robotB()
        elif self.robot_id == 102:
            self.robotC()
        elif self.robot_id == 103:
            self.robotD()
        elif self.robot_id == 104:
            self.robotE()
        else:
            print("Invalid robot id ", self.robot_id)
            assert False
        self.built = True
        return self.get_objects()

    def get_objects(self):
        assert self.built
        return self.objects, self.springs, self.faces

