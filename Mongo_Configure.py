import numpy as np
from pymongo import MongoClient
from scipy import sparse, io
import json

class DB_manager:
    client = MongoClient('localhost', 27017)
    db = client.test1
    training_data = db.training_datas

    def import_training_data(self, word_vector_file, train_label_file, vector_type_file):
        self.training_data.delete_many({})
        self.training_data.create_index('training_num')
        word_vector = io.mmread(word_vector_file)
        vector = word_vector.todense()
        with open(train_label_file, 'r') as f:
            label = json.load(f)
        with open(vector_type_file, 'r') as f:
            word = json.load(f)

        num = len(label)
        for i in range(num):
            dic = {}
            dic['training_num'] = i
            dic['vector'] = list(vector[i])
            dic['content'] = list(np.array(word)[vector[i]])
            dic['label'] = int(label[i])
            self.training_data.insert_one(dic)


if '__main__' == __name__:
    DB_manager().import_training_data('Data/word_vector.mtx', 'Data/train_label.json',
                                      'Data/vector_type.json')
