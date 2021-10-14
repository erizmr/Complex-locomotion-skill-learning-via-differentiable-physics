
if __name__ == "__main__":
    import os
    import argparse
    from multitask.robot_design import RobotDesignMassSpring, RobotDesignMPM

    parser = argparse.ArgumentParser(description='Check Robot Design')
    parser.add_argument('--robot_design_file',
                        default='',
                        help='robot design file')
    parser.add_argument('--solver',
                        default="mass_spring",
                        help='solver type')
    args = parser.parse_args()
    file_name = args.robot_design_file.split('/')[-1].split('.')[0]
    robot_design_file = args.robot_design_file
    if args.solver == "mass_spring":
        robot_builder = RobotDesignMassSpring.from_file(robot_design_file)
    elif args.solver == "mpm":
        robot_builder = RobotDesignMPM.from_file(robot_design_file)
    else:
        raise NotImplementedError(f"Solver {args.solver} not implemented.")
    id = robot_builder.robot_id
    # obj, spr = robot_builder.build()
    robot_builder.build()
    robot_builder.draw()

    current_path = os.path.dirname(os.path.realpath(__file__))
    robot_builder.dump_to_json(file=os.path.join(current_path, "robot_design_data/{}.json".format(file_name)))
