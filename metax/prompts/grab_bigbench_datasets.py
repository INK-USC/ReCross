import seqio
from bigbench.bbseqio import tasks
import tensorflow_datasets as tfds
import os
import json
from random import shuffle


BIG_BENCH_TASKS = [
    "code_line_description",
    "hindu_knowledge",
    "known_unknowns",
    "logic_grid_puzzle",
    "misconceptions",
    "movie_dialog_same_or_different",
    "novel_concepts",
    "strategyqa",
    "formal_fallacies_syllogisms_negation",
    "vitaminc_fact_verification",
    "winowhy",
    # "logical_deduction", # These two don't have input/target_pretokenized
    # "conceptual_combinations",
]


def get_save_dir(task_name):
    return f"data/{task_name}/"


def get_seqio_task_id(task_name):
    """seqio needs a string to access the task
    see https://github.com/google/BIG-bench/blob/main/bigbench/benchmark_tasks/seqio_task_catalog.md
    """
    return f"bigbench:{task_name}.mul.t5_default_vocab.0_shot.all_examples"


def generate_fewshot_data(formatted_data):
    # fewshot_data = {16:{}, 32:{}, 64:{}, 128:{}, 256:{}, 512:{}}
    # Since some datasets are small, we only do 16 shots
    fewshot_data = {16: {}}
    for round in range(20):
        shuffle(formatted_data)
        for shot in list(sorted(fewshot_data.keys())):
            fewshot_data[shot][f"round#{round}"] = formatted_data[:shot]
    return fewshot_data


def grab_data_of_task(task):
    print(f"Grabbing data of {task}")

    # Access dataset
    dataset = seqio.get_mixture_or_task(get_seqio_task_id(task)).get_dataset(
        sequence_length={"inputs": 512, "targets": 512}, split="all"
    )

    cnt = 0  # Used for id generation
    formatted_data = []  # List of instances in MetaCross format
    for dp in tfds.as_numpy(dataset):
        input = dp['inputs_pretokenized'].decode('utf-8')
        target = dp['targets_pretokenized'].decode('utf-8')
        id = f"{task}|all|default|#{cnt}"
        cnt += 1
        formatted_dp = [input, [target, ], id]
        formatted_data.append(formatted_dp)
    print(
        f"Done grabbing data of {task}. Have {len(formatted_data)} instances in total")

    fewshot_data = generate_fewshot_data(formatted_data)

    save_dir = get_save_dir(task)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Save fewshot data
    with open(f"{save_dir}fewshot.json", "w") as f:
        json.dump(fewshot_data, f, indent=4)

    # Save test data
    shuffle(formatted_data)
    with open(f"{save_dir}test-1000.json", "w") as f:
        # Notice here it's possible we have < 1000 examples
        # The filename is for consistency
        json.dump(formatted_data[:1000], f, indent=4)


def main():
    for task in BIG_BENCH_TASKS:
        grab_data_of_task(task)


if __name__ == "__main__":
    main()
