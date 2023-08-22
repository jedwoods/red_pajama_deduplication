import torch
from retext import get_all_files

# from tqdm import tqdm
# from transformers import GPT2Tokenizer, GPT2Model
import numpy as np
import json
from sentence_transformers import SentenceTransformer

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_tokens(
    file,
    limit=1000,
    page_id=1,
    array_id=0,
    model=SentenceTransformer("sentence-t5-xxl"),
):
    """Get tokens from a jsonl file and save them to a .npz file with x number of arrays of tokens where x is the limit defined."""

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    with open(file, "r") as infile:
        multiplier = 1000
        output_json = {page_id: {}}

        for line in infile:
            # outfile = f"tokens_{page_id}"

            rawdata = json.loads(line)
            rawlines = rawdata["sentences"]

            for raw_sentence in rawlines:
                array_id += 1
                sentence_embeddings = model.encode(raw_sentence)
                output_json[page_id][array_id] = np.array(sentence_embeddings)

                while array_id > limit:
                    # temp limit to make sure it isn't changed if in the process of catching an error
                    check_limit = int(limit)

                    # catching errors in memory allocation in the case that the limit is too high
                    try:
                        # saving the tokens to an npz file
                        for page in output_json:
                            np.savez(f"tokens_{page_id}", **output_json[page])

                        # resetting the output_json and increasing the limit
                        page_id += 1
                        output_json = {page_id: {}}
                        limit = limit + multiplier

                    # if the limit is too high, the limit is halved and the tokens are saved to two separate files
                    except Exception as e:
                        # catching the error and printing it
                        print(e)
                        print("Error in file: " + dir + "on line: " + line)
                        print("current limit: " + str(limit))

                        # halving the limit
                        multiplier = multiplier / 2

                        # defining the array ids for the two separate arrays
                        array_id_middle = check_limit - multiplier
                        array_id_lower = check_limit - multiplier * 2

                        # instantiates two separate arrays witht half the data each to write to smaller files
                        temp_1 = {
                            key: output_json[key]
                            for key in output_json[page_id]
                            if array_id_lower <= key < array_id_middle
                        }
                        page_id += 1
                        np.savez(f"tokens_{page_id}", **temp_1)
                        limit += multiplier

                        temp_2 = {
                            key: output_json[key]
                            for key in output_json[page_id]
                            if array_id_lower < key <= check_limit
                        }
                        page_id += 1
                        np.savez(f"tokens_{page_id}", **temp_2)
                        limit += multiplier

    return array_id, page_id, limit


if __name__ == "__main__":
    dir = "/mnt/pccfs2/not_backed_up/red_pajama/Actual files to use"
    files = get_all_files(dir)
    # page_id = 0
    # array_id = 0

    for file in files:
        array_id, page_id, limit = get_tokens(file=file)


"""to do:
- rewrite the token saving strcuture and np.savez with the end goal of saving the tokens from a scalable dictionary of tokens

- test the token saving structure"""
