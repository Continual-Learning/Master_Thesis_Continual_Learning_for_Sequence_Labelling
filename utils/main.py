import argparse

import all_commands.dataset_utils as dataset_cmd
import all_commands.postprocessing_utils as post_cmd

if __name__ == "__main__":
    HELP_STR = """Run one of several data_utils commands:
balance_dataset                     Create balanced datasets from unbalanced ones via random undersampling
    --input_dir   Path to directory containing the datasets to balance
    --output_dir  Path to output directory
merge_datasets                      Create single dataset from two by matching findings.
    --input_dir   Path to directory of first dataset
    --output_dir  Path to output directory
    Path to directory of second dataset and undersampling ratio of majority class will be prompted
split_dataset                       Split dataset into two parts of configurable size. Output in input directory
    --input_dir   Path to directory containing the datasets to balance
undersample_with_negative_mentions  Undersample majority class in dataset by keeping negative mentions
                                    (output by clinical review product) of target finding when undersampling.
    --input_dir   Path to directory containing the datasets to undersample
    --output_dir  Path to output directory
    Target dataset size and path to multi-label output of each dataset will be prompted.
undersample_no_conflicts            Similar to undersample_with_negative_mentions, but will also ensure that there are
                                    no conflicting mentions among findings.
    --input_dir   Path to directory containing the datasets to undersample
    --output_dir  Path to output directory
    Target dataset size and path to multi-label output of each dataset will be prompted.
create_parallel_corpus              Creates a parallel corpus from a dataset and the corresponding texts obtained
                                    via machine translation (split into training and dev datasets)
    --input_dir   Path to directory containing the original reports
    --output_dir  Path to output directory
    Path to translated datasets will be prompted
remove_optimizer_from_checkpoints   Removes optimizers from PyTorch checkpoints to reduce memory required by models
    --input_dir   Path to directory containing the original checkpoints
    --output_dir  Path to output directory
transfer_embeddings                 Prepares dataset to use machine translation technique + regularization.
                                    Should be used after extracting embeddings for the original corpus
                                    using the evaluation script.
    --input_dir   Path to directory containing the original reports
    --output_dir  Path to output directory
    Path to directory containing embeddings and translated reports will be prompted.
"""

    parser = argparse.ArgumentParser(description=HELP_STR,
                                     formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('--exec', '-e', required=True,
                        help="Command to execute")
    parser.add_argument('--input_dir', '-i', required=True, default=None,
                        help="Input directory")
    parser.add_argument('--output_dir', '-o', required=False, default=None,
                        help="output directory")

    args = vars(parser.parse_args())
    input_dir = args["input_dir"]
    out_dir = args["output_dir"]
    command = args["exec"]

    if command in ["balance_dataset",
                   "convert_epoch_evaluation",
                   "prepare_evaluation_of_epochs",
                   "merge_datasets",
                   "remove_optimizer_from_checkpoints",
                   "undersample_with_negative_mentions",
                   "undersample_no_conflicts",
                   "create_parallel_corpus"]:
        if not out_dir:
            out_dir = input("Output directory? ")

    if command == "balance_dataset":
        dataset_cmd.balance_dataset(input_dir, out_dir)
    elif command == "split_dataset":
        split_ratio = float(input("Split ratio? "))
        dataset_cmd.split_dataset(input_dir, split_ratio)
    elif command == "merge_datasets":
        second_dir = input("Second input directory? ")
        undersampling_1 = float(input("Undersampling ratio of first dataset? "))
        undersampling_2 = float(input("Undersampling ratio of second dataset? "))
        dataset_cmd.merge_datasets(input_dir, second_dir, undersampling_1, undersampling_2, out_dir)
    elif command == "undersample_with_negative_mentions" or command == "undersample_no_conflicts":
        target_size = int(input("Target dataset size? "))
        cr_outputs = input("Path to multi-label output file? ")
        getattr(dataset_cmd, command)(input_dir, target_size, cr_outputs, out_dir)
    elif command == "mimic_undersampling":
        target_size = int(input("Target dataset size? "))
        dataset_cmd.undersample_mimic_no_conflicts(input_dir, target_size)
    elif command == "create_parallel_corpus":
        second_dir = input("Translated data input directory? ")
        dataset_cmd.create_parallel_corpus(input_dir, second_dir, out_dir)
    elif command == "remove_optimizer_from_checkpoints":
        post_cmd.remove_optimizer_from_checkpoints(input_dir, out_dir)
    elif command == "transfer_embeddings":
        second_dir = input("Translated data input directory? ")
        embedding_dir = input("Embedding directory? ")
        id_column = input("Sample identifier column? ")
        post_cmd.transfer_embedding_to_parallel_corpus(input_dir, second_dir, embedding_dir, id_column)
