import json
import argparse


def json_load(prefix):
    return json.load(open("cfg/{}.json".format(prefix), "r"))


def json_dump(obj, prefix, script_file):
    print("for i in 1 2 3;", file=script_file)
    print("""do
python3 main_diff_phy.py --config_file cfg/{}.json --train
python3 main_diff_phy.py --config_file cfg/{}.json --evaluate --no-tensorboard-train --evaluate_path saved_results/{}/DiffTaichi_DiffPhy
done
""".format(prefix, prefix, prefix), file = script_file)

    return json.dump(obj, open("cfg/{}.json".format(prefix), "w"), indent=4)


# FULL: original reference, OP: Optimizer, AF: Activation Function, BS: Batch Size, PS: Periodic Signal, SV: StateVector, TG: Targets, LD: naive_loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Ablation Study')
    parser.add_argument('--prefix',
                        default="sim_config_DiffPhy_robot7_vhc_5_l01",
                        help='prefix')
    parser.add_argument('--output-name',
                        default="run_scripts_robot7_ablation.sh",
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
    args = parser.parse_args()

    script_file = open(args.output_name, "w")
    print("#!/bin/bash", file=script_file)
    prefix = args.prefix  # "sim_config_DiffPhy_robot7_vhc_5_l01"
    full_path = prefix

    if args.full:
        # FULL
        prefix_original = prefix
        full_json = json_load(full_path)
        json_dump(full_json, prefix_original, script_file)

    if args.slip and args.friction > 0:
        # slip boundary
        prefix_slip = prefix + "_slip" + f"_{args.friction}"
        full_json = json_load(full_path)
        full_json["simulator"]["friction"] = args.friction
        json_dump(full_json, prefix_slip, script_file)

    if args.op:
        # OP
        prefix_sgd = prefix + "_sgd"
        full_json = json_load(full_path)
        full_json["nn"]["optimizer"] = "sgd"
        full_json["nn"]["learning_rate"] = 1e-2
        json_dump(full_json, prefix_sgd, script_file)

    if args.af:
        # AF
        prefix_tanh = prefix + "_tanh"
        full_json = json_load(full_path)
        full_json["nn"]["activation"] = "tanh"
        json_dump(full_json, prefix_tanh, script_file)

    if args.bs:
        # BS
        prefix_bs = prefix + "_batch_1"
        full_json = json_load(full_path)
        full_json["nn"]["batch_size"] = 1
        json_dump(full_json, prefix_bs, script_file)

    if args.ps:
        # PS
        prefix_no_periodic = prefix + "_no_periodic"
        full_json = json_load(full_path)
        full_json["nn"]["n_sin_waves"] = 0
        json_dump(full_json, prefix_no_periodic, script_file)

    if args.sv:
        # SV
        prefix_no_state_vector = prefix + "_no_state_vector"
        full_json = json_load(full_path)
        full_json["nn"]["has_state_vector"] = 0
        json_dump(full_json, prefix_no_state_vector, script_file)

    if args.tg:
        # TG
        prefix_no_targets = prefix + "_no_targets"
        full_json = json_load(full_path)
        full_json["nn"]["duplicate_v"] = 0
        full_json["nn"]["duplicate_h"] = 0
        full_json["nn"]["duplicate_c"] = 0
        json_dump(full_json, prefix_no_targets, script_file)

    if args.ld:
        # LD
        prefix_no_targets = prefix + "_naive_loss"
        full_json = json_load(full_path)
        full_json["process"]["naive_loss"] = True
        json_dump(full_json, prefix_no_targets, script_file)

    if args.cq:
        # LD
        prefix_hu2019 = prefix + "_hu2019"
        full_json = json_load(full_path)
        full_json["nn"]["batch_size"] =1
        full_json["nn"]["optimizer"] = "sgd"
        full_json["nn"]["learning_rate"] = 1e-2
        json_dump(full_json, prefix_hu2019, script_file)




# OP: Optimizer, AF: Activation Function, BS: Batch Size, PS: Periodic Signal, SV: StateVector, TV: Targets, LD: naive_loss