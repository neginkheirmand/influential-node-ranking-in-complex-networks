import networkx as nx
import matplotlib.pyplot as plt
import os
import pandas as pd



directory = "./datasets/"
graph_list = []
for dirpath, _, files in os.walk(directory):
    for filename in files:
        try:
            if filename.endswith(".edges"):
                file_path = os.path.join(dirpath, filename) 
                graph_list.append(file_path)
                graph = nx.read_edgelist(file_path, comments="%", nodetype=int)
                b_list = sir_model(graph, 0.1,5,1000)
                b_dict = get_Bdict_from_Blist(b_list)
                i = 0
                for b in b_dict.keys():
                    b_dict[b]=sorted(b_dict[b], key=lambda x: x[0])
                    x_  = [t[0] for t in b_dict[b]]
                    y_  = [t[1] for t in b_dict[b]]
                    Sir = pd.DataFrame({'Node':x_,'SIR':y_})
                    Sir.to_csv(f'{filename}.csv',index=False)
                    i+=1
        except Exception as e: 
            print(e, f'{filename}')