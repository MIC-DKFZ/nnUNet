from batchgenerators.utilities.file_and_folder_operations import save_json
labels = {
    0: 'background',
    (1, 2, 3): 'whole tumor',
    (2, 3): 'tumor core',
    3: 'enhancing tumor'
}
save_json(labels, 'labels.json', sort_keys=False) # -> will crash

import yaml
with open('labels.yaml', 'w') as f:
    yaml.dump(labels, f)  # -> is ugly and not human readable


labels_inverted = {j: i for i, j in labels.items()}
save_json(labels_inverted, 'labels_inverted.json', sort_keys=False)  # -> works but unintuitive
