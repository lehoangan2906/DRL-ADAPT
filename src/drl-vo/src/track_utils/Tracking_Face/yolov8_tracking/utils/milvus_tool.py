from milvus import Milvus, IndexType, MetricType, Status
import pickle
import numpy as np

def top_k_pred(j,top_k,samples,results):
    names = []
    for i in range(top_k):
        names.append(samples[results[j][i].id]['name'])  
        # names.append(samples['folder'][results[j][i].id])   
    check = {}
    for name in names:
        check[name] = names.count(name)
    _max = 0.
    for key in check.keys():
        if check[key] > _max:
            _max = check[key]
            out_name = key
    
    score = []
    cnt = 0
    for i in range(top_k):
        if samples[results[j][i].id]['name'] == out_name:
            # distance = min(distance, float(results[j][i].distance)) #float(results[j][i].distance)
            score.append(results[j][i].distance)
            cnt += 1
    return out_name, score[0], None #samples['imagepath'][results[j][0].id]


def init_Milvus():
    client = Milvus(uri='tcp://localhost:19530')
    # client = Milvus(uri='tcp://172.21.100.254:19530')
    client.list_collections()

    client.drop_collection('data')

    # Create collection demo_collection if it dosen't exist.
    collection_name = 'data'

    status, ok = client.has_collection(collection_name)
    if not ok:
        param = {
            'collection_name': collection_name,
            'dimension': 512,
            'metric_type': MetricType.IP  # optional
        }
    client.create_collection(param)

    _, collection = client.get_collection_info(collection_name)
    print("Collection: ", collection)
    status, result = client.count_entities(collection_name)
    print("Result: ", result)

    pkl_path = "../sample_huy.pkl"
    with open(pkl_path, 'rb') as f:
        samples = pickle.load(f)
    names = [sample['name'] for sample in samples]
    names = set(names)

    # embs = np.array([np.array(i['emb']) for i in samples])
    embs = np.array([sample['emb'] for sample in samples])
    print(len(embs))
    print(len(samples))
    #%% convert list to numpy array
    knownEmbedding = embs
    print(knownEmbedding.shape)

    # knownNamesId = [i['name'] for i in samples]
    knownNamesId = [sample['name'] for sample in samples]
    print('len Ids: ',len(knownNamesId))
    print('len Embs: ',knownEmbedding.shape[0])
    # insert true data into true_collection_
    status, ids = client.insert(collection_name=collection_name, records=knownEmbedding, ids=list(range(len(knownNamesId))))
    if not status.OK():
        print("Insert failed: {}".format(status))
    print(len(ids))
    print('Status: ',status)
    #%%
    client.flush([collection_name])
    # Get demo_collection row count
    status, result = client.count_entities(collection_name)
    print(result)
    print(status) 
    ivf_param = {'nlist': 1024}
    status = client.create_index(collection_name, IndexType.FLAT, ivf_param)

    # describe index, get information of index
    status, index = client.get_index_info(collection_name)
    
    return client, collection_name, samples