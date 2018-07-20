import glob
import numpy as np
import pandas as pd
from math import sqrt
import queue as Q

TOP_K = 30
OUTPUT=open("Item_Result"+str(TOP_K)+".txt","w")
########################### STEP 1 ##############################
# import data
r_cols = ['user_id', 'item_id', 'rating']
train_filenames = glob.glob("ml-100k/*.base")
trains = []
for filename in train_filenames[0:5]:  # discard ua.base and ub.base
    trains.append(pd.read_csv(filename, sep='\t', names=r_cols,usecols = range(3),encoding='latin-1'))
    
test_filenames = glob.glob("ml-100k/*.test")
tests = []
for filename in test_filenames[0:5]:
    tests.append(pd.read_csv(filename, sep='\t', names=r_cols,usecols = range(3),encoding='latin-1'))

########################### STEP 2 ##############################
for u in range(5):
    train_ratings = trains[u]
    test_ratings = tests[u]
    #Calculating the mean rating and subtracting from each rating of a user to calculate the adjusted rating
    mean= train_ratings[['item_id','rating']].groupby(['item_id'], as_index = False, sort = False).mean().rename(columns = {'rating': 'rating_mean'})
    train_ratings = pd.merge(train_ratings,mean,on = 'item_id', how = 'left', sort = False)
    train_ratings['rating_adjusted']=train_ratings['rating']-train_ratings['rating_mean']
    SE = 0
    AE = 0
    #Compute simialrity between each 2 movies
    #To save runtime, only take 2000 test cases
    for i in range(0,test_ratings.shape[0],100):
        sum_predict = 0
        sum_similarity = 0
        t = test_ratings.loc[[i]]
        t_u = np.array(t["user_id"])[0]
        t_i = np.array(t["item_id"])[0]
        t_r = np.array(t["rating"])[0]
        # info about target movie in train dataset
        target_data = train_ratings[train_ratings['item_id']==t_i]
        target_data = target_data[['user_id','rating_adjusted']].drop_duplicates()
        target_data = target_data.rename(columns={'rating_adjusted':'rating_adjusted1'})
        target_val = np.sqrt(np.sum(np.square(target_data['rating_adjusted1'])))+0.5 # add 0.5 for smoothing
        # find all movies that this user have rated in training dataset
        rated = train_ratings[train_ratings['user_id']==t_u]
        rated = np.unique(rated["item_id"])
        # build an priority Queue to store TOP_K similar item
        result = Q.PriorityQueue()
        for m in rated:
            #Compute similarity between movie in test set(which user have not rated) with rated in training_data
            movie_data = train_ratings[train_ratings['item_id']==m]
            movie_data = movie_data[['user_id','rating_adjusted']].drop_duplicates()
            movie_data = movie_data.rename(columns={'rating_adjusted':'rating_adjusted2'})
            movie_val = np.sqrt(np.sum(np.square(movie_data['rating_adjusted2'])))+0.5 
            merge_data = pd.merge(target_data,movie_data,on = 'user_id', how = 'inner', sort = False)
            merge_data['dot'] = (merge_data['rating_adjusted1']*merge_data['rating_adjusted2'])
            sum_val = merge_data['dot'].sum() +0.25
            similarity = sum_val/(target_val*movie_val)
            # put (similarity, rating) into priority queue
            result.put((similarity,(np.array(train_ratings[(train_ratings['user_id']==t_u)&(train_ratings["item_id"]==m)]["rating"])[0])))
            if(len(result.queue)>TOP_K):
                result.get()
        # Compute predict value
        true_value = np.array(test_ratings.loc[[i]]["rating"])[0]
        for e in result.queue:
            sum_predict += e[0]*e[1]
            sum_similarity += e[0]
        predict = (sum_predict+0.5)/(sum_similarity+0.5)  #avoiding sum_similarity=0
        # Calculate square error
        SE += np.square(true_value - predict)
        AE += abs(true_value - predict)
    # Calculate root of mean square error
    RMSD = np.sqrt(SE/int(test_ratings.shape[0]/100))
    MAE = AE/int(test_ratings.shape[0]/100)
    # Write answer
    OUTPUT.write("U"+str(u+1)+"\tTOP_K: "+str(TOP_K)+"\n")
    OUTPUT.write("RMSD: "+str(RMSD))
    OUTPUT.write("\tMAE: "+str(MAE))
    OUTPUT.write("\n")

OUTPUT.close()
