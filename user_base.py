import glob
import numpy as np
import pandas as pd
from math import sqrt
import queue as Q

TOP_K = 30
OUTPUT=open("User_Result"+str(TOP_K)+".txt","w")
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
    mean= train_ratings[['user_id','rating']].groupby(['user_id'], as_index = False, sort = False).mean().rename(columns = {'rating': 'rating_mean'})
    train_ratings = pd.merge(train_ratings,mean,on = 'user_id', how = 'left', sort = False)
    train_ratings['rating_adjusted']=train_ratings['rating']-train_ratings['rating_mean']
    SE = 0
    AE = 0
    for i in range(0,test_ratings.shape[0],100):
        sum_predict = 0
        sum_similarity = 0
        t = test_ratings.loc[[i]]
        t_u = np.array(t["user_id"])[0]
        t_i = np.array(t["item_id"])[0]
        user1_data= train_ratings[train_ratings["user_id"]==t_u]
        user1_data=user1_data.rename(columns={"rating_adjusted":"rating_adjusted1"})
        user1_data=user1_data.rename(columns={"user_id":"user_id1"})
        user1_val=np.sqrt(np.sum(np.square(user1_data["rating_adjusted1"]), axis=0))+0.5 #0.5 for smoothing avoiding new user(who has not rated anyone yet)
        # find user who rated this movie
        distinct_users1=np.unique(train_ratings[train_ratings["item_id"]==t_i]["user_id"])
        result = Q.PriorityQueue()
        for user2 in distinct_users1:
            user2_data= train_ratings[train_ratings["user_id"]==user2]
            user2_data=user2_data.rename(columns={"rating_adjusted":"rating_adjusted2"})
            user2_data=user2_data.rename(columns={"user_id":"user_id2"})
            user2_val=np.sqrt(np.sum(np.square(user2_data["rating_adjusted2"]), axis=0))+0.5
            user_data = pd.merge(user1_data,user2_data[["rating_adjusted2","item_id","user_id2"]],on = "item_id", how = "inner", sort = False)
            user_data["dot"]=(user_data["rating_adjusted1"]*user_data["rating_adjusted2"])
            sum_val = user_data['dot'].sum()+0.25
            similarity = sum_val/(user1_val*user2_val)
            # put (similarity, rating) into priority queue
            result.put((similarity,(np.array(train_ratings[(train_ratings['user_id']==user2)&(train_ratings["item_id"]==t_i)]["rating"])[0])))
            if(len(result.queue)>TOP_K):
                result.get()
        # Compute predict value
        true_value = np.array(test_ratings.loc[[i]]["rating"])[0]
        for e in result.queue:
            sum_predict += e[0]*e[1]
            sum_similarity += e[0]
        predict = (sum_predict+0.5)/(sum_similarity+0.5)  # smoothing avoiding sum_similarity=0
        # Calculate square error
        SE += np.square(true_value - predict)
        AE += abs(true_value - predict)
    # Calculate root of mean square error
    RMSD = np.sqrt(SE/int(test_ratings.shape[0]/100))
    MAE = AE/int(test_ratings.shape[0]/100)
    # Write answer
    OUTPUT.write("U"+str(u)+" TOP_K: "+str(TOP_K))
    OUTPUT.write("RMSD: "+str(RMSD))
    OUTPUT.write("MAE: "+str(MAE))
    OUTPUT.write("")

OUTPUT.close()           

