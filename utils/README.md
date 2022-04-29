# Utils package

This package contains some utility functions meant to simplify dataset preparations, file conversions and output post-processing.

General usage (additional required input parameters may be prompted):
```bash
python3 -m main -e <command_name> -i <input_path> -o <output_path>
```

To get a short description of the commands that can be run within this subpackage execute:
```bash
python3 -m main -h
```

Most commands are dataset-independent, i.e. will work with any type of data as specified in the [main README](../README.md#Input-data-format).
Others, however, are project-specific and will only work with the outputs provided by certain tools, which are not included in this project.
These include `undersample_with_negative_mentions`, `undersample_no_conflicts`, `mimic_undersampling`.
Hence, we recommend using taylored undersampling methods for each custom dataset.