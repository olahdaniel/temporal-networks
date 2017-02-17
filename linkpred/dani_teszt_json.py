# argumentumok: json file
# "/mnt/idms/temporalNodeRanking/data/filtered_timeline_data/tsv/15o/15o_only_first_mentions_for_rec_0"
# "/mnt/idms/home/danielolah/predicted_links_teszt"
import sys
sys.path.insert(0,"../")

import pyrecsys.experiments as exp
import pandas as pd
import json
from pprint import pprint

if len(sys.argv) != 2:
    print("dani_teszt_json.py <json>")

with open(sys.argv[1]) as config_file:    
    parameters = json.load(config_file)

pprint(parameters)
    
data = pd.read_csv(parameters["input_file"],
    sep=' ',
    header=None,
    names=['time', 'user', 'item', 'score', 'eval']
)
print(len(data))
topkexp = exp.GlobalTopKExperiment(
    negativeRate=9,
    minTime=1318244928,
    fileName = parameters["output_file"] + ".gtl",
    topK = 1000
)

topkexp.run(data, outFile=parameters["output_file"] + ".out")
