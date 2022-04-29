import itertools
import json
import os

import pandas as pd
from sklearn.model_selection import train_test_split


def balance_dataset(input_dir, out_dir):
    for file in os.listdir(input_dir):
        if "_data.csv" not in file:
            continue

        dataset = pd.read_csv(os.path.join(input_dir, file))

        dataset_pos = dataset[dataset["label"] == 1]
        dataset_neg = dataset[dataset["label"] == 0]

        print(f"Original dataset size: {len(dataset)}: {len(dataset_pos)} positives, {len(dataset_neg)} negatives.")

        dataset_neg = dataset_neg.sample(n=len(dataset_pos))

        new_dataset = pd.concat([dataset_pos, dataset_neg])

        print(f"Size of balanced dataset: {len(new_dataset)}")

        new_dataset.to_csv(os.path.join(out_dir, file))


def split_dataset(input_dir, split_ratio):
    os.mkdir(os.path.join(input_dir, "split_1"))
    os.mkdir(os.path.join(input_dir, "split_2"))
    for file in os.listdir(input_dir):
        if "_data.csv" not in file:
            continue

        dataset = pd.read_csv(os.path.join(input_dir, file))

        dataset_pos = dataset[dataset["label"] == 1]
        dataset_neg = dataset[dataset["label"] == 0]

        print(f"Original dataset size: {len(dataset)}: {len(dataset_pos)} positives, {len(dataset_neg)} negatives.")

        split_1, split_2 = train_test_split(dataset, test_size=split_ratio, random_state=42)

        pos_1 = sum(split_1["label"])
        neg_1 = len(split_1) - pos_1
        print(f"Size of split one: {len(split_1)}, {pos_1} pos, {neg_1} neg")
        pos_2 = sum(split_2["label"])
        neg_2 = len(split_2) - pos_2
        print(f"Size of split one: {len(split_2)}, {pos_2} pos, {neg_2} neg")

        split_1.to_csv(os.path.join(input_dir, "split_1", file))
        split_2.to_csv(os.path.join(input_dir, "split_2", file))


def merge_datasets(path_1, path_2, undersampling_1, undersampling_2, out_dir):
    for file in os.listdir(path_1):
        if "_data.csv" not in file:
            continue

        filepath_1 = os.path.join(path_1, file)
        filepath_2 = os.path.join(path_2, file)

        df_1 = pd.read_csv(filepath_1)
        df_1_pos = df_1[df_1["label"] == 1]
        df_1_neg = df_1[df_1["label"] == 0].sample(frac=undersampling_1, random_state=42)
        df_1 = pd.concat([df_1_pos, df_1_neg])

        df_2 = pd.read_csv(filepath_2)
        df_2_pos = df_2[df_2["label"] == 1]
        df_2_neg = df_2[df_2["label"] == 0].sample(frac=undersampling_2, random_state=42)
        df_2 = pd.concat([df_2_pos, df_2_neg])

        df_1["origin"] = path_1
        df_2["origin"] = path_2

        merged_df = pd.concat([df_1, df_2])
        merged_df.to_csv(os.path.join(out_dir, file))


def __get_multi_label_outputs(multi_label_outputs):
    data = [row for row in open(multi_label_outputs, encoding='utf8').read().split("\n") if row != ""]

    assert len(data) % 2 == 0

    sliced_data = [data[i:i + 2] for i in range(0, len(data), 2)]
    assert list(itertools.chain.from_iterable(sliced_data)) == data

    filename_to_output = {}
    for filename, output in sliced_data:
        filename_to_output[filename.split("/")[-1]] = json.loads(output)
    return filename_to_output


def undersample_with_negative_mentions(input_dir, target_size, multi_label_outputs, out_dir):
    filename_to_output = __get_multi_label_outputs(multi_label_outputs)

    for file in os.listdir(input_dir):
        if "_data.csv" not in file:
            continue

        curr_finding = file.split("_data")[0]
        df = pd.read_csv(os.path.join(input_dir, file))
        mentions = []

        for filename in df["filename"]:
            found = False
            for insight in filename_to_output[filename]["insights"]:
                if insight["code"] == curr_finding:
                    mentions.append(1 if insight["observationValue"] == "Negative" else 0)
                    found = True
                    break

            if not found:
                mentions.append(0)

        assert(len(mentions) == len(df))
        df["mentions"] = pd.Series(mentions)

        df_neg = df[df["label"] == 0]
        df_pos = df[df["label"] == 1]

        max_majority_samples = target_size - len(df_pos)
        # Balance the dataset by also reducing negatives
        if max_majority_samples < len(df_pos):
            df_pos = df_pos.sample(n=target_size // 2, random_state=42)
            max_majority_samples = target_size - len(df_pos)  # Handles rare case of odd target size?

        # 1 corresponds to negative mention, 0 to positive mention or no mention
        negative_mentions = df_neg[df_neg["mentions"] == 1]
        no_mentions = df_neg[df_neg["mentions"] == 0]

        print(f"{curr_finding}: {len(negative_mentions)} negative mentions, "
              f"{len(no_mentions)} reports missing mention.")
        negative_mentions = negative_mentions.sample(n=min(len(negative_mentions), max_majority_samples//2),
                                                     random_state=42)
        no_mentions = no_mentions.sample(n=max_majority_samples-len(negative_mentions), random_state=42)

        print(f"{curr_finding} after undersampling: {len(negative_mentions)} negative mentions, "
              f"{len(no_mentions)} reports missing mention.")

        out_df = pd.concat([df_pos, negative_mentions, no_mentions])
        assert len(out_df) == target_size, "Could not reach target size with input data"

        out_df.to_csv(os.path.join(out_dir, file), index=False)


def undersample_no_conflicts(input_dir, target_size, multi_label_outputs, out_dir):
    filename_to_output = __get_multi_label_outputs(multi_label_outputs)

    findings = [file.split("_data.csv")[0] for file in os.listdir(input_dir) if "_data.csv" in file]

    for file in os.listdir(input_dir):
        if "_data.csv" not in file:
            continue

        curr_finding = file.split("_data")[0]
        df = pd.read_csv(os.path.join(input_dir, file))
        keep = []

        for filename in df["filename"]:
            pos_mention = False
            neg_mention = False
            pos_mention_of_other_finding = False
            for insight in filename_to_output[filename]["insights"]:
                if insight["code"] == curr_finding:
                    pos_mention = insight["observationValue"] == "Positive"
                    neg_mention = insight["observationValue"] == "Negative"
                elif insight["code"] in findings and insight["observationValue"] == "Positive":
                    pos_mention_of_other_finding = True

            if not pos_mention and not neg_mention and pos_mention_of_other_finding:
                keep.append(False)
            else:
                keep.append(True)

        df = df[keep]
        df_no_dupl = df.drop_duplicates(subset="text")
        print(f"Dropped {len(df)-len(df_no_dupl)} duplicates")
        df = df_no_dupl

        df.to_csv(os.path.join(out_dir, file), index=False)

    undersample_with_negative_mentions(input_dir, target_size, multi_label_outputs, out_dir)

    filenames_kept = set()
    for file in os.listdir(input_dir):
        if "_data.csv" not in file:
            continue
        df = pd.read_csv(os.path.join(input_dir, file))

        for filename in df["filename"]:
            filenames_kept.add(filename)

    train_filenames = pd.Series(list(filenames_kept))
    train_filenames = train_filenames.sample(frac=0.7)
    train_filenames = set(train_filenames)

    train_dir = os.path.join(out_dir, "train")
    eval_dir = os.path.join(out_dir, "eval")
    os.mkdir(train_dir)
    os.mkdir(eval_dir)

    for file in os.listdir(input_dir):
        if "_data.csv" not in file:
            continue

        curr_finding = file.split("_data")[0]
        df = pd.read_csv(os.path.join(out_dir, file))

        train_mask = []
        for filename in df["filename"]:
            train_mask.append(filename in train_filenames)

        train_df = df[train_mask]
        eval_df = df[[not mask for mask in train_mask]]

        print(f"Finding {curr_finding} split into {len(train_df)} training samples and "
              f"{len(eval_df)} validation samples.")

        train_df.to_csv(os.path.join(train_dir, file), index=False)
        eval_df.to_csv(os.path.join(eval_dir, file), index=False)


def undersample_unlabelled_no_conflicts(input_dir, target_dataset_size):
    out_dir = os.path.join(input_dir, "sampled_no_conflicts")
    os.mkdir(out_dir)

    datasets = {}
    n_pos = []

    for file in os.listdir(input_dir):
        if "_data.csv" not in file:
            continue

        finding = file.split("_")[0]
        df = pd.read_csv(os.path.join(input_dir, file))

        datasets[finding] = df
        n_pos.append((finding, len(df[df["pred"] > 0.5])))

    # Iterate over datasets from lowest number of positives
    n_pos.sort(key=lambda x: x[1])

    reserved_reports = set()
    for finding, _ in n_pos:
        df: pd.DataFrame = datasets[finding]
        df_pos = df[df["pred"] > 0.5]
        df_pos = df_pos.sample(n=min(target_dataset_size//2, len(df_pos)), random_state=42)
        # Sampling similar to when we have clinical review labels, except we do not have negative mentions
        # => Every negative is treated as a no-mention report and discarded from other datasets
        df_neg = df[df["pred"] < 0.5]
        df_neg = df_neg[~df_neg["filename"].isin(reserved_reports)]
        df_neg = df_neg.sample(n=min(target_dataset_size - len(df_pos), len(df_neg)), random_state=42)

        print(f"finding {finding}: {len(df_pos)} positives, {len(df_neg)} negatives")

        df_sampled = pd.concat([df_pos, df_neg])
        reserved_reports.update(df_neg["filename"])
        df_sampled.to_csv(os.path.join(out_dir, f"{finding}_data.csv"), index=False)

def create_parallel_corpus(en_dir, fr_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    for dataset in ["train", "eval"]:
        os.makedirs(os.path.join(out_dir, dataset))
        for file in os.listdir(os.path.join(en_dir, dataset)):
            if "_data.csv" in file:
                en_df = pd.read_csv(os.path.join(en_dir, dataset, file))
                fr_df = pd.read_csv(os.path.join(fr_dir, file))

                en_filenames = set(en_df["filename"])
                fr_parallel = [filename not in en_filenames for filename in fr_df["filename"]]

                fr_df = fr_df[fr_parallel]
                fr_df.to_csv(os.path.join(out_dir, dataset, file), index=False)
