import random
import math
from datetime import datetime

def randomized_search():
    hyperparameter_combination = []
    search_space = {"epochs" : [5,10],
                    "batch_size" : [2,4,8,16],
                    "gradient_accumulation" : [2,4,8,16]}

    n_iter = math.ceil(10/100*(len(search_space["epochs"])*len(search_space["batch_size"])*len(search_space["gradient_accumulation"])))

    for i in range(n_iter):
        param2, param3 = 32,32
        while param2*param3 > 32:
            param1 = random.choice(search_space["epochs"])
            param2 = random.choice(search_space["batch_size"])
            param3 = random.choice(search_space["gradient_accumulation"])
        x=datetime.now()
        current_datetime = str(x.strftime("%d"))+str(x.strftime("%b"))+str(x.strftime("%y"))
        out_dir = current_datetime + "/" + str(i)

        print([param1, param2, param3], out_dir)

if __name__ == '__main__':
    randomized_search()