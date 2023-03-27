
if __name__ == "__main__":
    import os
    import argparse
    from multitask.robot_design import RobotDesignMassSpring3D

    parser = argparse.ArgumentParser(description='Check Robot Design')
    parser.add_argument('--robot_design_file',
                        default='',
                        help='robot design file')
    args = parser.parse_args()
    file_name = args.robot_design_file.split('/')[-1].split('.')[0]
    robot_design_file = args.robot_design_file
    robot_builder = RobotDesignMassSpring3D.from_file(robot_design_file)
    id = robot_builder.robot_id
    # obj, spr = robot_builder.build()
    robot_builder.build()
    robot_builder.draw()

    # current_path = os.path.dirname(os.path.realpath(__file__))
    # robot_builder.dump_to_json(file=os.path.join(current_path, "robot_design_data/{}.json".format(file_name)))
