# from datasets import load_dataset
import json
from nltk.tokenize import sent_tokenize
import os

from tqdm import tqdm
from gensim.corpora import TextCorpus


class ChunkedTextCorpus(TextCorpus):
    def __init__(self, input, chunksize=1000, **kwargs):
        self.input = input
        self.chunksize = chunksize
        super().__init__(**kwargs)

    def get_texts(self):
        with open(self.input, "r") as file:
            while True:
                chunk = file.readlines(self.chunksize)
                if not chunk:
                    break
                yield [self.preprocess_text(text) for text in chunk]


def get_all_files(
    directory,
):
    """This function takes in a directory and returns a list of files in that directory, formatted as a dictionary of lists."""

    dirs1 = os.listdir(directory)
    list_set = {}

    for set in dirs1:
        # print("checkpoint")
        file_list = []
        list_set[set] = file_list
        # print(set)
        path1 = directory + "/" + set

        for root, dirs, files in os.walk(path1):
            # print("checkpoint2")
            # print(files)
            for file in files:
                # print(files)
                file_list.append(os.path.join(root, file))
            list_set[set] = file_list
    return list_set


def tokenize_rp(dir, infiles):
    count = 1

    # Specify the new directory
    directory = dir + "_sentence_files"
    new_dir = (
        "/mnt/pccfs2/not_backed_up/red_pajama/tokenized_sentences" + "/" + directory
    )

    # Create the directory
    if not os.path.exists(new_dir):
        os.mkdir(
            "/mnt/pccfs2/not_backed_up/red_pajama/tokenized_sentences" + "/" + directory
        )

    # scrape the files
    donefiles = os.walk(new_dir)

    scraped_files = infiles
    # for root, dirs, files in os.walk(dir):
    #     for file in files:
    #         # Obtain all files in the directory
    #         scraped_files.append(os.path.join(root, file))
    for file in tqdm(scraped_files):
        # testfile = os.path.join(new_dir, file)
        new_file_name = f"{os.path.basename(file)}%s.jsonl" % count
        count += 1
        with open(
            "/mnt/pccfs2/not_backed_up/red_pajama/tokenized_sentences"
            + "/"
            + directory
            + "/"
            + new_file_name,
            "w",
        ) as outfile:
            with open(file, "r") as json_file:
                # finish this tomorrow you scumbag
                line_count = 0
                chunk = 100
                storage = []
                data = {
                    "title": "",
                    "sentences": [],
                }
                # load the files
                for jsonl in json_file:
                    to_tokenize = json.loads(jsonl)

                    # pull out the text metadata
                    to_process = to_tokenize["text"]

                    # seperate into a list of sentences
                    sentences = sent_tokenize(to_process)

                    # store the sentences as json
                    data["sentences"] = sentences

                    storage.append(data)

                    # write the sentences to a file

                for line in storage:
                    line_count += 1
                    sents = line["sentences"]
                    for i in range(0, len(sents), chunk):
                        temp = {}

                        try:
                            temp[line_count] = sents[i : i + chunk]
                            json.dump(temp[line_count], outfile)
                        except OSError:
                            temp[line_count] = sents[i : i + chunk / 2]
                            json.dump(temp, outfile)
                            temp[line_count] = sents[i + chunk / 2 : i + chunk]
                            json.dump(temp, outfile)
                            print("OSError in file: " + file)
                            continue


if __name__ == "__main__":
    initial_directory = "/mnt/pccfs2/not_backed_up/red_pajama/Actual files to use"

    # this should return a dictionary of directories with a list of files in each directory
    dir = get_all_files(initial_directory)

    # print(dir)
    # for key in dir:
    #     print(key, ":", len(dir[key]))
    for key in dir:
        tokenize_rp(key, dir[key])
