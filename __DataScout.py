import pandas as pd
from chembl_webresource_client.new_client import new_client

def data_scout(target_input, size):
    """
    iterate through all targets and find out how many bioactivity data units we'll have to work with for each one
    \n it's better to choose an index containing a higher number since the model may not learn well from just 4-10 data points assuming those compounds make it through the filtering
    :return: target index: number of bioactivity data units
    """

    target = new_client.target
    target_query = target.search(target_input)
    targets = pd.DataFrame.from_dict(target_query)

    bioact_pt_list = {}

    for indx in range(len(targets)):
        selected_target = targets.target_chembl_id[indx]
        bioact_pt_list.update({indx:len(new_client.activity.filter(target_chembl_id=selected_target).filter(standard_type = "IC50"))})

    for dict_idx in data_sorted(bioact_pt_list, size):
        for idx, num_data in dict_idx.items():
            print(f"{idx}: {num_data}")

def data_sorted(bioact_dict, size):
    sorted_indices = sorted(bioact_dict.items(), key=lambda x:x[1], reverse=True)
    return [{indx: count} for indx, count in sorted_indices[:size]]
