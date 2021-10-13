objects = []
springs = []

def clear():
    objects = []
    springs = []

def add_object(x):
    objects.append(x)
    return len(objects) - 1


def add_spring(a, b, length=None, stiffness=1, actuation=0.1):
    if length == None:
        length = ((objects[a][0] - objects[b][0])**2 +
                  (objects[a][1] - objects[b][1])**2)**0.5
    springs.append([a, b, length, stiffness, actuation])


def robotA():
    add_object([0.2, 0.1])
    add_object([0.3, 0.13])
    add_object([0.4, 0.1])
    add_object([0.2, 0.2])
    add_object([0.3, 0.2])
    add_object([0.4, 0.2])

    s = 14000

    def link(a, b, actuation=0.1):
        add_spring(a, b, stiffness=s, actuation=actuation)

    link(0, 1)
    link(1, 2)
    link(3, 4)
    link(4, 5)
    link(0, 3)
    link(2, 5)
    link(0, 4)
    link(1, 4)
    link(2, 4)
    link(3, 1)
    link(5, 1)

    return objects, springs


points = []
point_id = []
mesh_springs = []


def add_mesh_point(i, j):
    if (i, j) not in points:
        id = add_object((i * 0.05 + 0.1, j * 0.05 + 0.1))
        points.append((i, j))
        point_id.append(id)
    return point_id[points.index((i, j))]


def add_mesh_spring(a, b, s, act):
    if (a, b) in mesh_springs or (b, a) in mesh_springs:
        return

    mesh_springs.append((a, b))
    add_spring(a, b, stiffness=s, actuation=act)


def add_mesh_square(i, j, actuation=0.0):
    a = add_mesh_point(i, j)
    b = add_mesh_point(i, j + 1)
    c = add_mesh_point(i + 1, j)
    d = add_mesh_point(i + 1, j + 1)

    # b d
    # a c
    add_mesh_spring(a, b, 3e4, actuation)
    add_mesh_spring(a, c, 3e4, 0)
    add_mesh_spring(a, d, 3e4, 0)
    add_mesh_spring(b, c, 3e4, 0)
    add_mesh_spring(b, d, 3e4, 0)
    add_mesh_spring(c, d, 3e4, actuation)


def add_mesh_triangle(i, j, actuation=0.0):
    a = add_mesh_point(i + 0.5, j + 0.5)
    b = add_mesh_point(i, j + 1)
    d = add_mesh_point(i + 1, j + 1)

    for i in [a, b, d]:
        for j in [a, b, d]:
            if i != j:
                add_mesh_spring(i, j, 3e4, 0)


def robotB(actuation=0.15):
    add_mesh_triangle(2, 0, actuation=actuation)
    add_mesh_triangle(0, 0, actuation=actuation)
    add_mesh_square(0, 1, actuation=actuation)
    add_mesh_square(0, 2)
    add_mesh_square(1, 2)
    add_mesh_square(2, 1, actuation=actuation)
    add_mesh_square(2, 2)
    # add_mesh_square(2, 3)
    # add_mesh_square(2, 4)

    return objects, springs


def robotC(actuation=0.2):
    add_mesh_square(2, 0, actuation=actuation)
    add_mesh_square(0, 0, actuation=actuation)
    add_mesh_square(0, 1)
    add_mesh_square(1, 1)
    add_mesh_square(2, 1)
    add_mesh_square(2, 2)
    add_mesh_square(2, 3)

    return objects, springs


def robotD(actuation=0.2):
    #add_mesh_square(2, 0, actuation=0.3)
    add_mesh_square(0, 0, actuation=actuation)
    add_mesh_square(0, 1, actuation=actuation)
    add_mesh_square(0, 2)
    add_mesh_square(1, 2)
    add_mesh_square(2, 1, actuation=actuation)
    add_mesh_square(2, 2)
    add_mesh_square(2, 3)
    add_mesh_square(2, 4)
    add_mesh_square(3, 1)
    add_mesh_square(4, 0, actuation=actuation)
    add_mesh_square(4, 1, actuation=actuation)

    return objects, springs


def robotE(actuation=0.2):
    add_mesh_square(0, 0, actuation=actuation)
    add_mesh_square(0, 1, actuation=actuation)
    add_mesh_square(0, 2)
    add_mesh_square(0, 3)
    add_mesh_square(1, 0, actuation=actuation)
    add_mesh_square(1, 1, actuation=actuation)
    add_mesh_square(1, 2)
    add_mesh_square(1, 3)
    add_mesh_square(2, 2)
    add_mesh_square(2, 3)
    add_mesh_square(3, 2)
    add_mesh_square(3, 3)
    add_mesh_square(4, 0, actuation=actuation)
    add_mesh_square(4, 1, actuation=actuation)
    add_mesh_square(4, 2)
    add_mesh_square(4, 3)
    add_mesh_square(5, 0, actuation=actuation)
    add_mesh_square(5, 1, actuation=actuation)
    add_mesh_square(5, 2)
    add_mesh_square(5, 3)

    return objects, springs


def robotF(actuation=0.3):
    # add_mesh_square(0, 0, actuation=0.15)
    # add_mesh_square(0, 1, actuation=0.15)
    # add_mesh_square(0, 2, actuation=0.15)
    # add_mesh_square(1, 2, actuation=0.15)
    # add_mesh_square(2, 2, actuation=0.15)
    # add_mesh_square(3, 0, actuation=0.15)
    # add_mesh_square(3, 1, actuation=0.15)
    # add_mesh_square(3, 2, actuation=0.15)
    # add_mesh_square(4, 2, actuation=0.15)
    # add_mesh_square(5, 2, actuation=0.15)
    # add_mesh_square(6, 0, actuation=0.15)
    # add_mesh_square(6, 1, actuation=0.15)
    # add_mesh_square(6, 2, actuation=0.15)

    add_mesh_square(0, 0, actuation=actuation)
    add_mesh_square(0, 1, actuation=actuation)
    add_mesh_square(1, 1, actuation=actuation)
    add_mesh_square(2, 1, actuation=actuation)
    add_mesh_square(3, 0, actuation=actuation)
    add_mesh_square(3, 1, actuation=actuation)

    return objects, springs

def add_mesh_square_big(x, y, actuation = 0.0):
    add_mesh_square(x * 2, y * 2, actuation)
    add_mesh_square(x * 2, y * 2 + 1, actuation)
    add_mesh_square(x * 2 + 1, y * 2, actuation)
    add_mesh_square(x * 2 + 1, y * 2 + 1, actuation)

def robotG(actuation=0.2):
    add_mesh_square_big(2, 0, actuation=actuation)
    add_mesh_square_big(0, 0, actuation=actuation)
    add_mesh_square_big(0, 1)
    add_mesh_square_big(1, 1)
    add_mesh_square_big(2, 1)
    add_mesh_square_big(2, 2)
    add_mesh_square_big(2, 3, actuation=actuation)

    return objects, springs

def robotH(actuation=0.2):
    add_mesh_square(2, 0, actuation=actuation)
    add_mesh_square(0, 0, actuation=actuation)
    add_mesh_square(0, 1)
    add_mesh_square(1, 1)
    add_mesh_square(2, 1)
    add_mesh_square(2, 2, actuation=actuation)
    add_mesh_square(2, 3)

    return objects, springs

robots = [robotA, robotB, robotC, robotD, robotE, robotF, robotG, robotH]
