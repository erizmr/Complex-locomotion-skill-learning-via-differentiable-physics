import json
import os

def load_from_json(str):
    items = json.loads(str)
    return items["solver"], items["objects"], items["springs"]

def current_path():
    return os.path.dirname(os.path.realpath(__file__))

def load_from_json_file(prefix):
    items = json.load(open(os.path.join(current_path(), "robot_design_data/{}.json".format(prefix)), "r"))
    return items["solver"], items["objects"], items["springs"]

def dump_to_json(solver, objects, springs, file = None):
    item = {"solver": solver, "objects": objects, "springs": springs}
    if file is None:
        str = json.dumps(item)
        return str
    json.dump(item, open(file, "w"), indent=4, sort_keys=False)


if __name__ == "__main__":
    import robot_config
    from multitask.robot_design import RobotDesignMassSpring
    robot_design_file = './robot_design/robot_2.json'
    robot = RobotDesignMassSpring.from_file(robot_design_file)
    robot_builders = []
    robot_builders.append(robot)

    for builder in robot_builders:
        id = builder.robot_id
        obj, spr = builder.build()
        builder.draw()
        builder.dump_to_json(file=os.path.join(current_path(), "robot_design_data/{}.json".format(id)))
        # dump_to_json("mass_spring", obj, spr, file=os.path.join(current_path(), "robot_design_data/{}.json".format(id)))
        robot_config.clear()
    
    # for id in range(5):
    #     print(load_from_json_file(id))