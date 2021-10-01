import json


def json_load(prefix):
    return json.load(open("cfg/{}.json".format(prefix), "r"))

script_file = open("run_scripts_robot7_abalation.sh", "w")
print("#!/bin/bash", file = script_file)

def json_dump(obj, prefix):
    print("for i in 1 2 3;", file=script_file)
    print("""do
python3 main_diff_phy.py --config_file cfg/{}.json --train
python3 main_diff_phy.py --config_file cfg/{}.json --evaluate --no-tensorboard-train --evaluate_path saved_results/{}/DiffTaichi_DiffPhy
done
""".format(prefix, prefix, prefix), file = script_file)

    return json.dump(obj, open("cfg/{}.json".format(prefix), "w"), indent=4)


prefix = "sim_config_DiffPhy_robot7_vhc_5_l01"

full_path = prefix

prefix_sgd = prefix + "_sgd"
full_json = json_load(full_path)
full_json["nn"]["optimizer"] = "sgd"
full_json["nn"]["learning_rate"] = 1e-2
json_dump(full_json, prefix_sgd)


prefix_tanh = prefix + "_tanh"
full_json = json_load(full_path)
full_json["nn"]["activation"] = "tanh"
json_dump(full_json, prefix_tanh)


prefix_no_periodic = prefix + "_no_periodic"
full_json = json_load(full_path)
full_json["nn"]["n_sin_waves"] = 0
json_dump(full_json, prefix_no_periodic)


prefix_no_state_vector = prefix + "_no_state_vector"
full_json = json_load(full_path)
full_json["nn"]["has_state_vector"] = 0
json_dump(full_json, prefix_no_state_vector)


prefix_no_targets = prefix + "_no_targets"
full_json = json_load(full_path)
full_json["nn"]["duplicate_v"] = 0
full_json["nn"]["duplicate_h"] = 0
full_json["nn"]["duplicate_c"] = 0
json_dump(full_json, prefix_no_targets)



# OP: Optimizer, AF: Activation Function, BS: Batch Size, PS: Periodic Signal, SV: StateVector, TV: Targets