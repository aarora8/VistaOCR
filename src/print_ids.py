import json
import sys
import os

if len(sys.argv) != 3:
    print("USAGE: python print_ids.py <data-dir> <output-prefix>")
    print("\tprints <output-prefix>-train.list, <output-prefix>-validation.list, <output-prefix>-test.list")
    sys.exit(1)


data_file = os.path.join(sys.argv[1], 'desc.json')
output_prefix = sys.argv[2]

with open(data_file, 'r') as fh:
    data = json.load(fh)

for split in ['train', 'validation', 'test']:
    with open(output_prefix + "-" + split + ".list", 'w') as fh:
        for entry in data[split]:
            uttid = entry['id']

            # Currently VistaOCR ids have stupid extra line id (0-indexed though!)
            # Strip it off for convinience
            uttid = uttid[ :uttid.rfind("_")]

            fh.write(uttid)
            fh.write("\n")
