
# coding: utf-8

# In[70]:

import gzip
from collections import defaultdict

def readGz(f):
  for l in gzip.open(f):
    yield eval(l)

file_read_counter = 0

alpha_array_train = []
alpha_array_test = []

outOf_train = []
outOf_test = []

nHelpful_train = []
nHelpful_test = []

ratings_train = []
ratings_test = []

wordCount_train = []
wordCount_test = []

for l in readGz("train.json.gz"):
    if file_read_counter < 100000:
        if l['helpful']['outOf'] != 0:
            
            nHelpful_train.append(l['helpful']['nHelpful'])
            outOf_train.append(l['helpful']['outOf'])
            alpha_array_train.append(float(l['helpful']['nHelpful'])/l['helpful']['outOf'])
            wordCount_train.append(len(l['reviewText'].split()))
            ratings_train.append(l['rating'])
            
    elif file_read_counter >= 900000:
        if l['helpful']['outOf'] != 0:
            nHelpful_test.append(l['helpful']['nHelpful'])
            outOf_test.append(l['helpful']['outOf'])
            alpha_array_test.append(float(l['helpful']['nHelpful'])/l['helpful']['outOf'])
            wordCount_test.append(len(l['reviewText'].split()))
            ratings_test.append(l['rating'])
            
    file_read_counter += 1

"""
# <Q1>
alpha  = sum(alpha_array_train)/len(alpha_array_train)
#0.737177206750307
errors = []
for i in range(len(outOf_test)):
    errors.append(abs(alpha*outOf_test[i] - nHelpful_test[i]))
mae = sum(errors)/len(errors)
#1.4160847148784765
# </Q1>
"""


#<Q2>
from scipy.optimize import minimize
def mse (a, helpfulness, word_count, rating):
    #helpfulness = nHelpful/outOf
    squared_error = 0
    for i in range(len(helpfulness)):
        squared_error += (helpfulness[i] - a[0] - a[1]*word_count[i] - a[2]*rating[i])**2
    mse = squared_error/len(helpfulness)
    return mse

alpha_array_train = numpy.array(alpha_array_train)
wordCount_train = numpy.array(wordCount_train)
ratings_train  = numpy.array(ratings_train)
res = minimize(mse, [0]*3, args=(alpha_array_train, wordCount_train, ratings_train), method='L-BFGS-B')
print res
"""
  status: 0
 success: True
    nfev: 68
     fun: 0.10574066882068517
       x: array([  4.58749120e-01,   1.42178581e-04,   5.97108339e-02])
 message: 'CONVERGENCE: NORM_OF_PROJECTED_GRADIENT_<=_PGTOL'
     jac: array([ -6.38378239e-08,   1.78052018e-06,  -1.01307851e-07])
     nit: 7

"""
a = res.x
errors = []
for i in range(len(alpha_array_test)):
    errors.append(abs(nHelpful_test[i] - (a[0] + a[1]*wordCount_test[i] + a[2]*ratings_test[i])*outOf_test[i]))
mae = sum(errors)/len(errors)
print mae
# 1.3007499216543448
#</Q2>


# In[98]:

#<Q4>
import gzip
from collections import defaultdict

def readGz(f):
  for l in gzip.open(f):
    yield eval(l)

### Rating baseline: compute averages for each user, or return the global average if we've never seen the user before

userItemPrediction = defaultdict(list)

userIds = []
itemIds = []
outOfs = []
wordCounts = []
ratings = []

for l in readGz("helpful.json.gz"):
    user,item,outOf,rating,wordCount = l['reviewerID'],l['itemID'],l['helpful']['outOf'],l['rating'],len(l['reviewText'].split())
    outOfs.append(outOf)
    userIds.append(user)
    itemIds.append(item)
    wordCounts.append(wordCount)
    ratings.append(rating)

a = res.x
userItemPrediction = defaultdict(list)

for i in range(len(userIds)):
    userItemPrediction[userIds[i]+'-'+itemIds[i]+'-'+str(outOfs[i])] = outOfs[i]*(a[0]+a[1]*wordCounts[i]+a[2]*ratings[i])

f_out = open('output.txt', 'w')
f = open('pairs_Helpful.txt', 'r')
f_out.write(f.readline())
for l in f.readlines():
    f_out.write(l.rstrip('\n')+','+str(userItemPrediction[l.rstrip('\n')])+'\n')
f_out.close()
f.close()

#Kaggle user name/Id: apoorvedave/458740 
#</Q4>


# In[125]:

#<Q5>
rating_alpha = sum(ratings_train)/len(ratings_train)
#4.0894368581424221


# In[5]:

#<Q6>
file_read_counter = 0
users_train = []
items_train = []
ratings_train = []

users_test = []
items_test = []
ratings_test = []

user_item_ratings_train = {}
item_user_ratings_train = {}
user_item_ratings_test = {}
count = 0
for l in readGz("train.json.gz"):
    if file_read_counter < 100000:
        users_train.append(l['reviewerID'])
        items_train.append(l['itemID'])
        ratings_train.append(l['rating'])
    elif file_read_counter >= 900000:
        users_test.append(l['reviewerID'])
        items_test.append(l['itemID'])
        ratings_test.append(l['rating'])
            
    file_read_counter += 1
    if file_read_counter %10000 == 0:
        print file_read_counter


# In[7]:

for i in range(len(users_train)):
    if users_train[i] not in user_item_ratings_train:
        user_item_ratings_train[users_train[i]] = {}
    user_item_ratings_train[users_train[i]][items_train[i]] = ratings_train[i]
    if items_train[i] not in item_user_ratings_train:
        item_user_ratings_train[items_train[i]] = {}
    item_user_ratings_train[items_train[i]][users_train[i]] = ratings_train[i]


# In[88]:

alpha = 1
set_users = set(users_train)
set_items = set(items_train)

list_users = list(set_users)
list_items = list(set_items)
lam = 1

beta_u = defaultdict(float)
beta_i = defaultdict(float)

for i in range(len(set_users)):
    beta_u[list_users[i]] = 0
for i in range(len(set_items)):
    beta_i[list_items[i]] = 0

old_alpha = 10

for i in range(len(set_users)):
    beta_u[list_users[i]] = 0
for i in range(len(set_items)):
    beta_i[list_items[i]] = 0
old_alpha = 10

while abs(alpha - old_alpha) > 0.001:
    old_alpha = alpha
    alpha = 0
    for i in range(len(ratings_train)):
        alpha += ratings_train[i] - beta_u[users_train[i]] - beta_i[items_train[i]]
    alpha = alpha/len(ratings_train)

    for user in beta_u:
        beta_u[user] = 0
        for item in user_item_ratings_train[user]:
            beta_u[user] += user_item_ratings_train[user][item] - alpha - beta_i[item]
        beta_u[user] = beta_u[user]/(lam+len(user_item_ratings_train[user]))

    for item in beta_i:
        beta_i[item] = 0
        for user in item_user_ratings_train[item]:
            beta_i[item] += item_user_ratings_train[item][user] - alpha - beta_u[user]
        beta_i[item] = beta_i[item]/(lam + len(item_user_ratings_train[item]))


# In[71]:

def mse_converge(lam):
    global alpha, beta_u, beta_i,users_train,items_train,ratings_train, user_item_ratings_train 
    alpha = 0
    for i in range(len(set_users)):
        beta_u[list_users[i]] = 0
    for i in range(len(set_items)):
        beta_i[list_items[i]] = 0
    old_alpha = 10
    
    while abs(alpha - old_alpha) > 0.001:
        old_alpha = alpha
        alpha = 0
        for i in range(len(ratings_train)):
            alpha += ratings_train[i] - beta_u[users_train[i]] - beta_i[items_train[i]]
        alpha = alpha/len(ratings_train)

        for user in beta_u:
            beta_u[user] = 0
            for item in user_item_ratings_train[user]:
                beta_u[user] += user_item_ratings_train[user][item] - alpha - beta_i[item]
            beta_u[user] = beta_u[user]/(lam+len(user_item_ratings_train[user]))

        for item in beta_i:
            beta_i[item] = 0
            for user in item_user_ratings_train[item]:
                beta_i[item] += item_user_ratings_train[item][user] - alpha - beta_u[user]
            beta_i[item] = beta_i[item]/(lam + len(item_user_ratings_train[item]))


# In[87]:

m=-400
u=''
for k in beta_i:
    if m < beta_i[k]:
        m = beta_i[k]
        u = k
m,u


# In[77]:

mse_converge(1)


# In[28]:

errors = []
for i in range(len(ratings_train)):
    errors.append(abs(ratings_train[i] - alpha - beta_u[users_train[i]] - beta_i[items_train[i]])**2)
sum(errors)/len(errors)

errors = []
for i in range(len(ratings_test)):
    errors.append(abs(ratings_test[i] - alpha - beta_u[users_test[i]] - beta_i[items_test[i]])**2)
sum(errors)/len(errors)
#MSE for training Set = 0.3335585596679398
#MSE for test Set = 0.9042631835049043
#alpha 4.2003778800008265


# In[32]:

sum(errors)/len(errors)


# In[75]:

errors = []
for i in range(len(ratings_test)):
    errors.append(abs(ratings_test[i] - alpha - beta_u[users_test[i]] - beta_i[items_test[i]])**2)
print sum(errors)/len(errors)


# In[63]:

alpha


# In[68]:

import scipy.optimize
lam = 1
def find_lam(lam):
    mse_converge(lam)
    return valid_mse()
x = scipy.optimize.minimize(find_lam,[1],args=(),method="BFGS")



# In[76]:


def model_predict(i,u):
    return (alpha + beta_u[u] + beta_i[i] )

predictions = open("predictions_Ratings.txt", 'w')

for l in open("pairs_Rating.txt"):
    if l.startswith("userID"):
        predictions.write(l)
        continue
    u,i = l.strip().split('-')
    pred = model_predict(i,u)
    predictions.write(u + '-' + i + ',' + str(pred) + '\n')

predictions.close()

#0.838915319212


# In[89]:

alpha


# In[90]:

errors = []
for i in range(len(ratings_test)):
    errors.append((ratings_test[i] - alpha - beta_u[users_test[i]] - beta_i[items_test[i]])**2)


# In[91]:

sum(errors)/len(errors)


# In[ ]:



