objects = []
springs = []
faces = []


def add_object(x):
    objects.append(x)
    return len(objects) - 1


def add_spring(a, b, length=None, stiffness=1, actuation=0.1):
    if length == None:
        length = ((objects[a][0] - objects[b][0])**2 +
                  (objects[a][1] - objects[b][1])**2 +
                  (objects[a][2] - objects[b][2])**2)**0.5
    springs.append([a, b, length, stiffness, actuation])

points = []
point_id = []
mesh_springs = []

def add_mesh_point(i, j, k):
    if (i, j, k) not in points:
        id = add_object((i * 0.05 + 0.1, j * 0.05 + 0.1, k * 0.05 + 0.1))
        points.append((i, j, k))
        point_id.append(id)
    return point_id[points.index((i, j, k))]

def add_foot_point(i, j, k):
    if (i, j, k) not in points:
        id = add_object((i * 0.05 + 0.125, j * 0.05 + 0.1, k * 0.05 + 0.125))
        points.append((i, j, k))
        point_id.append(id)
    return point_id[points.index((i, j, k))]

def add_mesh_spring(a, b, s, act):
    if (a, b) in mesh_springs or (b, a) in mesh_springs:
        return

    mesh_springs.append((a, b))
    add_spring(a, b, stiffness=s, actuation=act)


def add_mesh_square(i, j, k, actuation=0.0):
    a = add_mesh_point(i, j, k)
    b = add_mesh_point(i, j + 1, k)
    c = add_mesh_point(i + 1, j, k)
    d = add_mesh_point(i + 1, j + 1, k)
    e = add_mesh_point(i, j, k + 1)
    f = add_mesh_point(i, j + 1, k + 1)
    g = add_mesh_point(i + 1, j, k + 1)
    h = add_mesh_point(i + 1, j + 1, k + 1)

    # b d
    # a c
    add_mesh_spring(a, b, 3e4, actuation)
    add_mesh_spring(c, d, 3e4, actuation)
    add_mesh_spring(e, f, 3e4, actuation)
    add_mesh_spring(g, h, 3e4, actuation)

    add_mesh_spring(b, d, 3e4, 0)
    add_mesh_spring(a, c, 3e4, 0)
    add_mesh_spring(f, h, 3e4, 0)
    add_mesh_spring(e, g, 3e4, 0)

    add_mesh_spring(b, f, 3e4, 0)
    add_mesh_spring(d, h, 3e4, 0)
    add_mesh_spring(a, e, 3e4, 0)
    add_mesh_spring(c, g, 3e4, 0)

    add_mesh_spring(b, g, 3e4, 0)
    add_mesh_spring(d, e, 3e4, 0)
    add_mesh_spring(f, c, 3e4, 0)
    add_mesh_spring(h, a, 3e4, 0)

    def append_square_face(a, b, c, d):
        faces.append((a, b, c))
        faces.append((d, a, c))
    append_square_face(a, b, d, c)
    append_square_face(b, f, h, d)
    append_square_face(f, e, g, h)
    append_square_face(e, a, c, g)
    append_square_face(h, g, c, d)
    append_square_face(b, a, e, f)

def add_foot(i, j, k, actuation=0.0):
    x = add_foot_point(i, j, k)
    a = add_mesh_point(i, j + 1, k)
    b = add_mesh_point(i, j + 1, k + 1)
    c = add_mesh_point(i + 1, j + 1, k)
    d = add_mesh_point(i + 1, j + 1, k + 1)
    add_mesh_spring(x, a, 3e4, actuation)
    add_mesh_spring(x, b, 3e4, actuation)
    add_mesh_spring(x, c, 3e4, actuation)
    add_mesh_spring(x, d, 3e4, actuation)
    add_mesh_spring(a, b, 3e4, 0)
    add_mesh_spring(a, c, 3e4, 0)
    add_mesh_spring(b, d, 3e4, 0)
    add_mesh_spring(c, d, 3e4, 0)

    faces.append((x, a, c))
    faces.append((x, b, a))
    faces.append((x, d, b))
    faces.append((x, c, d))
    faces.append((a, b, d))
    faces.append((a, d, c))

def robotA():
    add_mesh_square(0, 0, 0, actuation=0.3)
    add_mesh_square(2, 0, 2, actuation=0.3)
    add_mesh_square(0, 0, 2, actuation=0.3)
    add_mesh_square(2, 0, 0, actuation=0.3)

    add_mesh_square(0, 1, 0, actuation=0)
    add_mesh_square(0, 1, 1, actuation=0)
    add_mesh_square(0, 1, 2, actuation=0)
    add_mesh_square(1, 1, 0, actuation=0)
    add_mesh_square(1, 1, 1, actuation=0)
    add_mesh_square(1, 1, 2, actuation=0)
    add_mesh_square(2, 1, 0, actuation=0)
    add_mesh_square(2, 1, 1, actuation=0)
    add_mesh_square(2, 1, 2, actuation=0)

    return objects, springs, faces

def robotB():
    add_foot(0, 0, 0, actuation=0.1)
    add_foot(0, 0, 1, actuation=0.1)
    add_foot(1, 0, 0, actuation=0.1)
    add_foot(1, 0, 1, actuation=0.1)

    return objects, springs, faces

def robotC():
    add_mesh_square(0, 0, 0, actuation=0.3)
    add_mesh_square(0, 1, 0, actuation=0)

    return objects, springs, faces

robots3d = [ robotA, robotB, robotC ]
