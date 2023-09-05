import torch
from retext import get_all_files

# from tqdm import tqdm
# from transformers import GPT2Tokenizer, GPT2Model
import numpy as np
import json
from sentence_transformers import SentenceTransformer

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def sentence_generator(sentences):
    for sentence in sentences:
        yield sentence

def check_limits(limit, current):
    if current == limit:
        status = True
    else:
        status = False
    return status


def get_tokens(
    file,
    limit=1000,
    page_id=1,
    array_id=0,
    model=SentenceTransformer("sentence-t5-xxl"),
):
    """Get tokens from a jsonl file and save them to a .npz file with x number of arrays of tokens where x is the limit defined."""

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # file should be formatted as a jsonl file with each line containing a json object with a "sentences" key containing a list of sentences
    with open(file, "r") as infile:
        # setting the limit and page_id
        multiplier = 1000

        # defing the output json which contains the two dictionaries, the first with the ID for the ouput page and the second with the ID for the array and the tokens
        output_json = {page_id: {}}

        for line in infile:
            # loading the json object
            rawdata = json.loads(line)

            # getting the list of sentences
            rawlines = rawdata["sentences"]

            line_id = 0 # the id of the current line of the files used for surplus lines that don't reach the limit

            # iterating through the sentences
            for raw_sentence in rawlines:
                # getting the sentence and increasing the line id
                line_id += 1

                status = check_limits(len(rawlines), line_id)
                
                # increasing the array id and adding the sentence embeddings to the output json
                array_id += 1
                sentence_embeddings = model.encode(raw_sentence)
                output_json[page_id][array_id] = np.array(sentence_embeddings)


                # setting a condition to save the tokens to a file if the limit is reached
                while array_id > limit: # and status == True:
                    # temp limit to make sure it isn't changed if in the process of catching an error
                    check_limit = int(limit)

                    # catching errors in memory allocation in the case that the limit is too high
                    try:
                        """The goal of this block is to see if the loading of the npz file will cause an error. 
                        If it does, the limit is halved and the tokens are saved to two separate files. The limit is then increased by the multiplier."""
                        # saving the tokens to an npz file with the page_id as the name and the array ID as keys
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

                        # halving the multiplier for increasing the limit
                        multiplier = multiplier / 2

                        # defining the array ids for the two separate arrays
                        array_id_middle = check_limit - multiplier
                        array_id_lower = check_limit - multiplier * 2

                        # instantiates two separate arrays with half the data each to write to smaller files
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
                            if array_id_middle <= key <= limit
                        }
                        # 
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
