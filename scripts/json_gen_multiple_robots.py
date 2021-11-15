import json
import argparse


def json_load(prefix):
    return json.load(open("cfg/{}.json".format(prefix), "r"))


def json_dump(obj, prefix, script_file, args):
    print("for i in 1 2 3;", file=script_file)
    print(f"""do
python3 main_diff_phy.py --config_file cfg/{prefix}.json --train --gpu-id {args.gpu_id} --memory {args.memory_train} --random
python3 main_diff_phy.py --config_file cfg/{prefix}.json --evaluate --gpu-id {args.gpu_id} --memory {args.memory_validation} --no-tensorboard-train --evaluate_path saved_results/{prefix}/DiffTaichi_DiffPhy
done
""", file=script_file)

    return json.dump(obj, open("cfg/{}.json".format(prefix), "w"), indent=4)


# FULL: original reference, OP: Optimizer, AF: Activation Function, BS: Batch Size, PS: Periodic Signal, SV: StateVector, TG: Targets, LD: naive_loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Ablation Study')
    parser.add_argument('--prefix',
                        default="sim_config_DiffPhy_robot{}_vhc_5_l01",
                        help='prefix')
    # parser.add_argument('--robots',
    #                     default="sim_config_DiffPhy_robot{}_vhc_5_l01",
    #                     help='prefix')
    parser.add_argument('--robots', '--list', nargs='+', help='selected robots', required=True)
    parser.add_argument('--output-name',
                        default="run_scripts_multiple_robots.sh",
                        help='output name')
    parser.add_argument('--full',
                        action='store_true',
                        help='full')
    parser.add_argument('--op',
                        action='store_true',
                        help='op')
    parser.add_argument('--af',
                        action='store_true',
                        help='op')
    parser.add_argument('--bs',
                        action='store_true',
                        help='op')
    parser.add_argument('--ps',
                        action='store_true',
                        help='op')
    parser.add_argument('--sv',
                        action='store_true',
                        help='op')
    parser.add_argument('--tg',
                        action='store_true',
                        help='op')
    parser.add_argument('--ld',
                        action='store_true',
                        help='op')
    parser.add_argument('--cq',
                        action='store_true',
                        help='cq')
    parser.add_argument('--slip',
                        action='store_true',
                        help='slip')
    parser.add_argument('--friction',
                        type=float,
                        default=-1.0,
                        help='friction coefficient')
    parser.add_argument('--activation-compare',
                        action='store_true',
                        help='do activation comparing')
    parser.add_argument('--activation-functions',
                        nargs='+',
                        help="all activation functions")
    parser.add_argument('--activation-functions-output',
                        nargs='+',
                        help="all activation functions for output layer")
    parser.add_argument('--adam-grid-search',
                        action='store_true',
                        help='do adam momentum grid search')
    parser.add_argument('--adam-b1',
                        nargs='+',
                        # default=0.90,
                        help="adam b1")
    parser.add_argument('--adam-b2',
                        nargs='+',
                        # default=0.90,
                        help="adam b2")
    parser.add_argument(
                        '--gpu-id',
                        default="1",
                        help='select gpu id')
    parser.add_argument(
                        '--memory-train',
                        type=float,
                        default=2.0,
                        help='pre allocated memory for taichi when training')
    parser.add_argument(
                        '--memory-validation',
                        type=float,
                        default=8.0,
                        help='pre allocated memory for taichi when evaluating')
    args = parser.parse_args()

    script_file = open(args.output_name, "w")
    print("#!/bin/bash", file=script_file)
    prefixes = [args.prefix.format(i) for i in args.robots]  # "sim_config_DiffPhy_robot7_vhc_5_l01"
    print(prefixes)

    for prefix in prefixes:
        full_path = prefix
        if args.full:
            # FULL
            prefix_original = prefix
            full_json = json_load(full_path)
            json_dump(full_json, prefix_original, script_file, args)

        if args.slip and args.friction > 0:
            # slip boundary
            prefix_slip = prefix + "_slip" + f"_{args.friction}".replace('.', "")
            full_json = json_load(full_path)
            full_json["simulator"]["friction"] = args.friction
            json_dump(full_json, prefix_slip, script_file, args)

        if args.op:
            # OP
            prefix_sgd = prefix + "_sgd"
            full_json = json_load(full_path)
            full_json["nn"]["optimizer"] = "sgd"
            full_json["nn"]["learning_rate"] = 1e-2
            json_dump(full_json, prefix_sgd, script_file, args)

        if args.af:
            # AF
            prefix_tanh = prefix + "_tanh"
            full_json = json_load(full_path)
            full_json["nn"]["activation"] = "tanh"
            json_dump(full_json, prefix_tanh, script_file, args)

        if args.bs:
            # BS
            prefix_bs = prefix + "_batch_1"
            full_json = json_load(full_path)
            full_json["nn"]["batch_size"] = 1
            json_dump(full_json, prefix_bs, script_file, args)

        if args.ps:
            # PS
            prefix_no_periodic = prefix + "_no_periodic"
            full_json = json_load(full_path)
            full_json["nn"]["n_sin_waves"] = 0
            json_dump(full_json, prefix_no_periodic, script_file, args)

        if args.sv:
            # SV
            prefix_no_state_vector = prefix + "_no_state_vector"
            full_json = json_load(full_path)
            full_json["nn"]["has_state_vector"] = 0
            json_dump(full_json, prefix_no_state_vector, script_file, args)

        if args.tg:
            # TG
            prefix_no_targets = prefix + "_no_targets"
            full_json = json_load(full_path)
            full_json["nn"]["duplicate_v"] = 0
            full_json["nn"]["duplicate_h"] = 0
            full_json["nn"]["duplicate_c"] = 0
            json_dump(full_json, prefix_no_targets, script_file, args)

        if args.ld:
            # LD
            prefix_no_targets = prefix + "_naive_loss"
            full_json = json_load(full_path)
            full_json["process"]["naive_loss"] = True
            json_dump(full_json, prefix_no_targets, script_file, args)

        if args.cq:
            # LD
            prefix_hu2019 = prefix + "_hu2019"
            full_json = json_load(full_path)
            full_json["nn"]["batch_size"] = 1
            full_json["nn"]["optimizer"] = "sgd"
            full_json["nn"]["learning_rate"] = 1e-2
            json_dump(full_json, prefix_hu2019, script_file, args)

        if args.adam_grid_search:
            full_json = json_load(full_path)
            adam_b1 = []
            adam_b2 = []
            if args.adam_b1 is not None:
                adam_b1 = args.adam_b1
            if args.adam_b2 is not None:
                adam_b2 = args.adam_b2

            for b_1 in adam_b1:
                prefix_adam_grid_search = prefix + "_adam_grid_search_b1_" + b_1.replace(".", "_")
                full_json["nn"]["adam_b1"] = float(b_1)
                for b_2 in adam_b2:
                    prefix_adam_grid_search_b2 = prefix_adam_grid_search + "_b2_" + b_2.replace(".","_")
                    full_json["nn"]["adam_b2"] = float(b_2)
                    json_dump(full_json, prefix_adam_grid_search_b2, script_file, args)
                if len(adam_b2) == 0:
                    json_dump(full_json, prefix_adam_grid_search, script_file, args)

        if args.activation_compare:
            full_json = json_load(full_path)
            for af in args.activation_functions:
                prefix_activation = prefix + "_activation_compare_" + af
                full_json["nn"]["activation"] = af
                for af_output in args.activation_functions_output:
                    full_json["nn"]["activation_output"] = af_output
                    prefix_activation = prefix_activation + "_" + af_output
                    json_dump(full_json, prefix_activation, script_file, args)


# OP: Optimizer, AF: Activation Function, BS: Batch Size, PS: Periodic Signal, SV: StateVector, TV: Targets, LD: naive_loss