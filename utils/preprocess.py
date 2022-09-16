import pandas as pd
from pathlib import Path
from argparse import ArgumentParser


"""
Proprocess the Vocal Burst data
"""

parser = ArgumentParser(description="File Splits")
parser.add_argument("--partition", type=str, default="train", choices=["Train", "Val", "Test"], help="partition to extract")
parser.add_argument("--data_info", type=str, help="Path to data_info file")
parser.add_argument("--type_info", type=str, help="Path to type info file")
parser.add_argument("--out_dir", type=str, help="Folder to output file to")

def create_split(partition:str, out_dir:Path, data_info=None, type_info=None):

    print("\n" + str(partition))

    # all the fields how they will appear in the final file
    columns = ["File_ID", "Voc_Type", "Country", "Valence", "Arousal", 
    "Awe", "Excitement", "Amusement", "Awkwardness", "Fear", "Horror", "Distress", "Triumph", "Sadness", "Surprise", 
    "China_Awe", "China_Excitement", "China_Amusement", "China_Awkwardness", "China_Fear", "China_Horror", "China_Distress" , "China_Triumph", "China_Sadness", "China_Surprise",
    "United States_Awe", "United States_Excitement" , "United States_Amusement", "United States_Awkwardness", "United States_Fear", "United States_Horror", "United States_Distress", "United States_Triumph", "United States_Sadness", "United States_Surprise",
    "South Africa_Awe","South Africa_Excitement","South Africa_Amusement","South Africa_Awkwardness", "South Africa_Fear", "South Africa_Horror", "South Africa_Distress","South Africa_Triumph", "South Africa_Sadness", "South Africa_Surprise",
    "Venezuela_Awe", "Venezuela_Excitement", "Venezuela_Amusement", "Venezuela_Awkwardness", "Venezuela_Fear", "Venezuela_Horror", "Venezuela_Distress", "Venezuela_Triumph", "Venezuela_Sadness", "Venezuela_Surprise"
    ]
    # number of decimal places to round the numerical fields to.
    precision = 6

    # open files

    data = pd.read_csv(str(data_info), sep=",", header="infer", low_memory=False)
    type_data = pd.read_csv(str(type_info), sep=",", header="infer")
    # merge data
    whole_data = pd.merge(data, type_data, on=["File_ID", "Split"])

    assert len(data) == len(whole_data)
    print("There are {} audio file ids".format( len(whole_data)))

    # filter for partition
    part_data = whole_data[whole_data["Split"] == str(partition)]
    print(len(part_data))

    # change column order and drop the Split Column as it contains no useful information now
    part_data = part_data[columns]

    # remove brackets from file id column
    part_data["File_ID"] = part_data["File_ID"].str.replace("\[", "", regex=True)
    part_data["File_ID"] = part_data["File_ID"].str.replace("\]", "", regex=True)

    # sort the frame
    part_data = part_data.sort_values("File_ID")

    # output file
    out_path = out_dir / (partition.lower() + ".csv")

    if partition == "Test":
        # replace NaN with empty string for export
        part_data.fillna("", inplace=True)
        part_data.to_csv(str(out_path), index=False)

    else:
         # round the dataframe columns that are numerical
        decimals = {c: precision for c in columns[3:]}
        part_data = part_data.round(decimals=decimals)

        part_data.to_csv(str(out_path), index=False)


# create file splits - one csv per partition
if __name__ == "__main__":

    args = parser.parse_args()
    partition = args.partition
    data_info = Path(args.data_info)
    type_info = Path(args.type_info)
    out_dir = Path(args.out_dir)
    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)

    create_split(partition, out_dir, data_info, type_info)

