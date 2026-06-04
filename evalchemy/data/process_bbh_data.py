from datasets import load_dataset, get_dataset_config_names, concatenate_datasets
import json

configs = get_dataset_config_names("lukaemon/bbh")

all_examples = []
for config in configs:
    print(f"Loading {config}...")
    ds = load_dataset("lukaemon/bbh", config)
    
    # Add task name to each example
    for split_name, split_data in ds.items():
        for example in split_data:
            example['task'] = config
            example['split'] = split_name
            all_examples.append(example)

print(f"\n✓ Total examples across all tasks: {len(all_examples)}")
output_file = "bbh_all_tasks.json"
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(all_examples, f, indent=2, ensure_ascii=False)
