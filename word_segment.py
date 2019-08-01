import os
import time
import csv
import ckipnlp.ws
ws = ckipnlp.ws.CkipWs(logger=False)


MAKEUP_ROOT = "/share/home/jiayu/project/cosmel/pushClean"
GOSSIP_ROOT = "/share/home/jiayu/project/ptt-scrapy/ptt/"


def read_message(ROOT_PATH, filename_head):
    """ Read csv file from files in path, return message and counts.
        Parameter:
            filename_head(str): like 'pushNoHttp_'.
        Return:
            message(dict: key=message, value=count)
    """
    print("Reading files...")
    message = {}
    
    filenames = [file for file in os.listdir(ROOT_PATH) if file.startswith(filename_head)]
    
    for filename in filenames:
        filepath = os.path.join(ROOT_PATH, filename)
        with open(filepath, newline='') as csvfile:
            rows = csv.DictReader(csvfile)

            for row in rows:
                if (row['message'] not in message.keys()):
                    message[row['message']] = 1
                else:
                    message[row['message']] += 1
    print("Finished reading files. %d unique messages read." % len(message.keys()))
    return message

def word_segment(text):
    """ Use CKIP WS to segment input sentence, returns a list.
    """
    text_ws = []
    
    try:
        temp = [line.strip() for line in ws(text).split('\n') if line.strip()!='']
        line_parse = [item.word for line in temp for item in ckipnlp.util.ws.WsSentence.from_text(line)]
    except:
        print("Error! Input text:", text, ", text length =", len(text))
    
    return line_parse


def write_ws_file(msg_count_dict, output_filename):
    print("Start writing file... ")
    with open(output_filename, "a") as writer:
        line_id = 0
        tStart = time.time()
        for line, count in msg_count_dict.items():
            if(line_id % 100000==0):
                tEnd = time.time()
                print("%d items to do word segment, %d items done word segment. Costs %f secends." 
                      % (len(msg_count_dict), line_id, tEnd-tStart))
            
            if(line.strip() != ""):
                line_ws = word_segment(line)
                for letter_id, letter in enumerate(line_ws):
                    if(letter_id < len(line_ws)-1):
                        print(letter, end=' ', file=writer)
                    else:
                        print(letter, end='\t', file=writer)
                print(count, file=writer)
            line_id += 1
    
    print("Finished reading files.")
    


def main():
    
#     makeup_msg_count = read_message(MAKEUP_ROOT, "pushClean_")
    gossip_msg_count = read_message(GOSSIP_ROOT, "pushGConnect_")
    
#     write_ws_file(makeup_msg_count, "./data/ptt_makeup_ws.tsv")
    write_ws_file(gossip_msg_count, "./data/ptt_gossip_ws.tsv")    

if __name__ == "__main__":
    main()
