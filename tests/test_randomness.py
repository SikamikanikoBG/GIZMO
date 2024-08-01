import json
from prettytable import PrettyTable


# def compare_pipeline_runs(run1_checkpoints, run2_checkpoints):
#     for cp1, cp2 in zip(run1_checkpoints, run2_checkpoints):
#         compare_checkpoints(cp1, cp2)

def compare_dicts(dict1, dict2, dict3):
    all_keys = set(dict1.keys()) | set(dict2.keys()) | set(dict3.keys())
    table = PrettyTable()
    table.field_names = ["Key", "Run 1 🚀", "Run 2 🚀", "Run 3 🚀"]
    table.align["Key"] = "l"
    table.align["Run 1 🚀"] = "l"
    table.align["Run 2 🚀"] = "l"
    table.align["Run 3 🚀"] = "l"

    for key in sorted(all_keys):
        value1 = json.dumps(dict1.get(key, "N/A"), indent=2)
        value2 = json.dumps(dict2.get(key, "N/A"), indent=2)
        value3 = json.dumps(dict3.get(key, "N/A"), indent=3)

        if value1 != value2:
            value1 = f"\033[91m{value1}\033[0m"  # Red color for differences
            value2 = f"\033[91m{value2}\033[0m"
        #
        if value1 != value3:
            value1 = f"\033[91m{value1}\033[0m"  # Red color for differences
            value3 = f"\033[91m{value3}\033[0m"
        #
        if value2 != value3:
            value2 = f"\033[91m{value2}\033[0m"  # Red color for differences
            value3 = f"\033[91m{value3}\033[0m"

        table.add_row([key, value1, value2, value3])

    print(table)

if __name__ == '__main__':
    run1_checkpoints_dir = "unittest/run_checkpoints_data_prep/run1.json"
    run2_checkpoints_dir = "unittest/run_checkpoints_data_prep/run2.json"
    run3_checkpoints_dir = "unittest/run_checkpoints_data_prep/run3.json"
    from collections import OrderedDict
    with open(run1_checkpoints_dir, 'r') as json_file:
        run1_checkpoints = json.load(json_file)

    with open(run2_checkpoints_dir, 'r') as json_file:
        run2_checkpoints = json.load(json_file)

    with open(run3_checkpoints_dir, 'r') as json_file:
        run3_checkpoints = json.load(json_file)

    for i in range(len(run1_checkpoints)):
        # Compare the i-th dictionary from each run
        compare_dicts(run1_checkpoints[i], run2_checkpoints[i], run3_checkpoints[i])