import numpy as np
from pymongo import MongoClient
from scipy import sparse, io
import json

class DB_manager:
    client = MongoClient()
    db = client.test1
    training_data = db.training_datas

    def import_training_data(self, word_vector_file, train_label_file):
        self.training_data.delete_many({})
        self.training_data.create_index('training_num')
        word_vector = io.mmread(word_vector_file)
        vector = np.array(word_vector.todense())
        with open(train_label_file, 'r') as f:
            label = json.load(f)

        num = len(label)
        for i in range(num):
            dic = {}
            dic['training_num'] = i
            dic['vector'] = list(vector[i])
            dic['label'] = int(label[i])
            self.training_data.insert_one(dic)


if '__main__' == __name__:
    DB_manager().import_training_data('Data/word_vector.mtx', 'Data/train_label.json')
