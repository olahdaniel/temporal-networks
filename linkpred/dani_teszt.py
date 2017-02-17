# argumentumok: input, output
# "/mnt/idms/temporalNodeRanking/data/filtered_timeline_data/tsv/15o/15o_only_first_mentions_for_rec_0"
# "/mnt/idms/home/danielolah/predicted_links_teszt"
import sys
sys.path.insert(0,"../")

import pyrecsys.experiments as exp
import pandas as pd

if len(sys.argv) != 3:
    print "dani_teszt.py <input> <output>"

data = pd.read_csv(sys.argv[1],
    sep=' ',
    header=None,
    names=['time', 'user', 'item', 'score', 'eval']
)
print(len(data))
topkexp = exp.GlobalTopKExperiment(
    negativeRate=9,
    minTime=1318244928,
    fileName = sys.argv[2] + ".gtl"
)

topkexp.run(data, outFile=sys.argv[2] + ".out")
