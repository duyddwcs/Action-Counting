import os
import json
import pandas as pd
'''
This script is generate a json splition for the MMFit dataset.
'''

train_dir = './MMFit/Train/'
val_dir = './MMFit/Test/'
idx = 0
data_dict = {}
train_data_dict = {}
val_data_dict = {}

query_length_list = []
exemplar_length_list = []
# Train
for action in os.listdir(train_dir):
    action_dir = os.path.join(train_dir, action)
    for ppl in os.listdir(action_dir):
        ppl_action_dir = os.path.join(action_dir, ppl)
        exemplar_dir = os.path.join(ppl_action_dir, 'Session 1')
        all_query_dir = os.path.join(ppl_action_dir, 'Session 2')
        exemplar_name = os.listdir(exemplar_dir)[0]
        exemplar_dir = os.path.join(exemplar_dir, exemplar_name)
        exemplar_count = exemplar_name.split('.csv')[0].split('_')[1]
        print('e', exemplar_count)
        for query in os.listdir(all_query_dir):
            query_dir = os.path.join(all_query_dir, query)
            query_count = query.split('.csv')[0].split('_')[1]
            print('q', query_count)
            query_dir = query_dir.split('./')[1]
            q_exemplar_dir = exemplar_dir.split('./')[1]
            idx += 1
            train_data_dict[idx] = {}
            train_data_dict[idx]['ExemplarPath'] = q_exemplar_dir
            train_data_dict[idx]['ExemplarCount'] = exemplar_count
            train_data_dict[idx]['QueryPath'] = query_dir
            train_data_dict[idx]['QueryCount'] = query_count
            train_data_dict[idx]['Action'] = action
            train_data_dict[idx]['Actor'] = ppl
            '''
            query_np = pd.read_csv(query_dir, header=None)
            query_np = query_np.to_numpy()
            query_length_list.append(query_np.shape[0])

            exemplar_np = pd.read_csv(q_exemplar_dir, header=None)
            exemplar_np = exemplar_np.to_numpy()
            exemplar_length_list.append(exemplar_np.shape[0])
            '''
#print(max(query_length_list))
#print(max(exemplar_length_list))


# Val
query_length_list = []
exemplar_length_list = []
for action in os.listdir(val_dir):
    action_dir = os.path.join(val_dir, action)
    for ppl in os.listdir(action_dir):
        ppl_action_dir = os.path.join(action_dir, ppl)
        exemplar_dir = os.path.join(ppl_action_dir, 'Session 1')
        all_query_dir = os.path.join(ppl_action_dir, 'Session 2')
        exemplar_name = os.listdir(exemplar_dir)[0]
        exemplar_dir = os.path.join(exemplar_dir, exemplar_name)
        exemplar_count = exemplar_name.split('.csv')[0].split('_')[1]
        print('e', exemplar_count)
        for query in os.listdir(all_query_dir):
            query_dir = os.path.join(all_query_dir, query)
            query_count = query.split('.csv')[0].split('_')[1]
            print('q', query_count)
            idx += 1
            val_data_dict[idx] = {}
            val_data_dict[idx]['ExemplarPath'] = exemplar_dir
            val_data_dict[idx]['ExemplarCount'] = exemplar_count
            val_data_dict[idx]['QueryPath'] = query_dir
            val_data_dict[idx]['QueryCount'] = query_count
            val_data_dict[idx]['Action'] = action
            val_data_dict[idx]['Actor'] = ppl
            '''
            query_np = pd.read_csv(query_dir, header=None)
            query_np = query_np.to_numpy()
            query_length_list.append(query_np.shape[0])

            exemplar_np = pd.read_csv(exemplar_dir, header=None)
            exemplar_np = exemplar_np.to_numpy()
            exemplar_length_list.append(exemplar_np.shape[0])
            '''

data_dict['Train'] = train_data_dict
data_dict['Val'] = val_data_dict

#print(max(query_length_list))
#print(max(exemplar_length_list))

with open("Train_val_split.json", "w") as outfile:
    json.dump(data_dict, outfile)