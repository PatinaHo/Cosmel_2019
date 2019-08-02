# Couldn't understand why???
#
# File "load_article.py", line 84
#     print(data[num], file=writer)
#                          ^
# SyntaxError: invalid syntax


import json
import os
import re
import subprocess
import csv

ROOT_PATH = "/share/home/patina/styleme/article/purged_article"
one_product_article_PATH = "/share/home/patina/styleme/article/one_product_article/"
one_prod_func_sent_PATH = "/share/home/patina/styleme/article/one_product_article/func_sentence"

def get_one_product_articles(jsonfile_PATH):
    """ Read in jsonfile exported from database; return a dictionary with key: article_id, value: product_id.
        return:
            one_product_article(set): {'9284': '16246', '5671': '11542', ...}
    """
    one_prod_article = {}
    with open(jsonfile_PATH) as file:
        one_prod_article_json = json.load(file)
    for item in one_prod_article_json[2]['data']:
        one_prod_article[item["article_id"]] = item["id"]

    return one_prod_article


def read_cat_func(jsonfile_PATH):
    with open(jsonfile_PATH) as json_file:
        data = json.load(json_file)
    return data


def read_prod_cat(jsonfile_PATH):
    with open(jsonfile_PATH) as json_file:
        data_list = json.load(json_file)[2]["data"]
    data = {}
    for prod in data_list:
        id = prod['id']
        type = prod['type']
        data[id] = type
    return data


def copy_articles(startRoot_PATH, dirs, dest_PATH, one_product_article, max_article_num):
    """Copy articles which matches the one_product limitation to destination directory.
    """
    # Walk through one_prod_articles
    copy_article = 0

    print("Start copying one_product_article files to destination ...")
    for directory in dirs:
        dir_path = os.path.join(startRoot_PATH, directory)
        for root, dirs, files in os.walk(dir_path):
            files = [f for f in files if not f[0] == '.']
            for file in files:
                if(copy_article < max_article_num):
                    file_path = os.path.join(dir_path, file)
                    article_id = re.split('_|\.', file)[1]
                    if(article_id in one_prod_article.keys()):
                        subprocess.run(["cp", file_path, dest_PATH])
                        copy_article += 1
    print("Finished copying.")
    
    return None



def extract_sent(one_product_article_PATH, output_PATH, filename, cat_func, prod_cat, article_prod):
    """ Extract sentence with key attributes and save as new file.
        Input:
            cat_func(dict): 
    """
    file_path       = os.path.join(one_product_article_PATH, filename)
    outputFile_path = os.path.join(output_PATH, filename)
    writer = open(outputFile_path, "w")

    with open(file_path) as file:
        data = file.readlines()
        data = [sent.strip() for sent in data]

    for num, line in enumerate(data):
        article_id = filename.split('_')[1][:-4]
        product_id = article_prod[article_id]
        category   = prod_cat[product_id]
        if (any(func in line for func in cat_func[category])):
            print(data[num], file=writer)

    writer.close()


def main():
    one_prod_article = get_one_product_articles(jsonfile_PATH="/share/home/patina/styleme/article/one_product_article/one_product_article.json")
    dirs = ['part-00000', 'part-00001', 'part-00002', 'part-00003', 'part-00004', 'part-00005', 'part-00006', 'part-00007', 'part-00008', 'part-00009']
    copy_articles(startRoot_PATH=ROOT_PATH, dirs=dirs, dest_PATH=one_product_article_PATH, one_product_article=one_prod_article, max_article_num=100)
    cat_func = read_cat_func("/share/home/patina/styleme/data/cat_func.json")
    prod_cat = read_prod_cat("/share/home/patina/styleme/data/product_category.json")
    for file in os.listdir(one_product_article_PATH):
        if file.endswith('.txt'):
            extract_sent(one_product_article_PATH=one_product_article_PATH, output_PATH=one_prod_func_sent_PATH, filename=file, cat_func=cat_func, prod_cat=prod_cat, article_prod=one_prod_article)

if __name__ == "__main__":
    main()