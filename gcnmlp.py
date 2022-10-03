from layers import Hetero2MLPPredictor
from util import *
import dgl.nn as dglnn
import pandas as pd
import dgl
import dgl.nn as dglnn
import torch.nn as nn
import dgl.function as fn
import torch.nn.functional as F
import numpy as np
from dgl.data.utils import generate_mask_tensor
# from dgl.data import DGLDataset
import torch
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score,precision_recall_curve, auc, f1_score,precision_score, recall_score,average_precision_score,label_ranking_average_precision_score
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder , LabelBinarizer
from sklearn import preprocessing


df = pd.read_csv("side-effect-and-drug_name.tsv",sep = "\t")
drug_id = df["drugbank_id"]
drug_name = df["drugbank_name"] 
side_effect =df["side_effect_name"] 
edgelist1 = zip(side_effect, drug_name)
dfs = df[['drugbank_name','side_effect_name']]

col_names = ["left_side","right_side","similairity"]
drugsim = pd.read_csv("semantic_similarity_side_effects_drugs.txt",sep ="\t",
                 names =col_names, header=None)
source =drugsim["left_side"]
destination = drugsim["right_side"]
similarity = drugsim["similairity"]
###Drugs similarity Network#####
edge_list = zip(source,destination,similarity)


le = preprocessing.LabelEncoder()
 
le.fit(dfs['drugbank_name'])
dfs['drug_id']=le.transform(dfs['drugbank_name']) 
drugsim['drug_id_left']=le.transform(drugsim['left_side'])
drugsim['drug_id_right']=le.transform(drugsim['right_side'])
 
le.fit(dfs['side_effect_name'])
dfs['se_id']=le.transform(dfs['side_effect_name'])
 
drug_id = torch.LongTensor(dfs['drug_id'])
side_id = torch.LongTensor(dfs['se_id'])

 
src = torch.LongTensor(drugsim['drug_id_left']) 
dst = torch.LongTensor(drugsim['drug_id_right'])   

 # Build graph
G = dgl.heterograph({
    ('drug_id', 'relate', 'side_id'): (drug_id, side_id),
    ('side_id', 'relate-by', 'drug_id'): (side_id, drug_id),
    ('drug_id','similar','drug_id'):(src,dst),
    ('drug_id','similar-by','drug_id'):(dst,src)
})

edge_weight2 = torch.tensor(drugsim['similairity'].to_numpy()) # drug-drug similar
G.edges['similar'].data['sim'] = edge_weight2
G.edges['similar-by'].data['sim'] = edge_weight2

u,v=G.edges(etype='relate')

eids = np.arange(G.number_of_edges(('drug_id', 'relate', 'side_id'))) # put id on edges ('drug_id', 'relate', 'side_id')

adj = sp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy()))) 

adj_neg = 1 - adj.todense() 
neg_u, neg_v = np.where(adj_neg != 0) 

class RGCN(nn.Module): 
    def __init__(self, in_feats, hid_feats, out_feats, rel_names): # in=10,hid=20,out=5,
        super().__init__()
        # featureless embedding
        embed_dict = {ntype : nn.Parameter(torch.Tensor(G.number_of_nodes(ntype), in_feats)) # in=10
                      for ntype in G.ntypes}
        for key, embed in embed_dict.items():
            nn.init.xavier_uniform_(embed)
        self.embed = nn.ParameterDict(embed_dict)
        # create layer
        self.conv1 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(in_feats, hid_feats) # 10,20
            for rel in rel_names}, aggregate='sum')
        
        self.conv2 = dglnn.HeteroGraphConv({
          rel: dglnn.GraphConv(hid_feats, hid_feats) # 20,20
          for rel in rel_names}, aggregate='sum')
                
        self.conv3 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(hid_feats, out_feats) # 20,5
            for rel in rel_names}, aggregate='sum')

    def forward(self, graph):
        # inputs are features of nodes
        h = self.conv1(graph, self.embed)
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv2(graph, h)
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv3(graph, h)
        return h


 


fprd = dict()
tprd = dict()
roc_aucd = dict()
precd = dict()
recalld = dict()
auc_prd = dict()



class Model(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, rel_names):
        super().__init__()
        self.sage = RGCN(in_features, hidden_features, out_features, rel_names)
        self.pred = Hetero2MLPPredictor(out_features, 1)
    def forward(self, g, pos_g,neg_g, etype):
        h = self.sage(g) # process incomplete G, output information
        pred_pos_g = self.pred(pos_g, h, etype) # use information inferring pos_g graph
        pred_neg_g = self.pred(neg_g, h, etype) # use information inferreing neg_g graph
        return torch.sigmoid(pred_pos_g), torch.sigmoid(pred_neg_g)





neg_eids = np.random.choice(len(neg_u), G.number_of_edges(('drug_id', 'relate', 'side_id')))  

FOLDS = 10
sz = G.number_of_edges(('drug_id', 'relate', 'side_id'))
eids = np.arange(sz) 
eids_pm = eids
test_size = fsz = int(sz /10) 
np.random.shuffle(eids_pm)
neg_eids_pm = neg_eids

IDX = u,v
IDX_neg = neg_u,neg_v
offset = 0
AUC_roc_train = np.zeros(FOLDS)
AUC_roc_valid = np.zeros(FOLDS)
AUC_roc_test = np.zeros(FOLDS)
AUC_pr_train = np.zeros(FOLDS)
AUC_pr_valid = np.zeros(FOLDS)
AUC_pr_test = np.zeros(FOLDS)
F1_micro_test = np.zeros(FOLDS)
F1_macro_test = np.zeros(FOLDS)
F1_weight_test = np.zeros(FOLDS)
Prec_micro_test = np.zeros(FOLDS)
Prec_macro_test = np.zeros(FOLDS)
Prec_weight_test  = np.zeros(FOLDS)
Recall_micro_test = np.zeros(FOLDS)
Recall_macro_test = np.zeros(FOLDS)
Recall_weight_test = np.zeros(FOLDS) 

algod = {}

for f in range(FOLDS):
    print("== Fold:",f," ==")
    test_pos_u, test_pos_v = idx_test = u[eids_pm[offset:offset+fsz]], v[eids_pm[offset:offset+fsz]]
    test_neg_u, test_neg_v = idx_test_neg = neg_u[neg_eids_pm[offset:offset+fsz]], neg_v[neg_eids_pm[offset:offset+fsz]]


    # training set
    train_pos_u, train_pos_v = u[eids_pm[np.r_[:offset,offset + fsz:len(eids_pm)]]], v[eids_pm[np.r_[:offset,offset + fsz:len(eids_pm)]]]
    # print('length of train_pos ',len(train_pos_u))
    train_neg_u, train_neg_v = neg_u[neg_eids_pm[np.r_[:offset,offset + fsz:len(neg_eids_pm)]]], neg_v[neg_eids_pm[np.r_[:offset,offset + fsz:len(neg_eids_pm)]]]
    # print('length of train_neg ',len(train_neg_u))

    # sub Graph: train_pos_g
    num_nodes_dict = {'drug_id': 1020, 'side_id': 5599}
    train_pos_g = dgl.heterograph({
        ('drug_id', 'relate', 'side_id'): (train_pos_u, train_pos_v),
        ('side_id', 'relate-by', 'drug_id'): (train_pos_v, train_pos_u),
        ('drug_id','similar','drug_id'):(src,dst),
        ('drug_id','similar-by','drug_id'):(dst,src)
        
    },num_nodes_dict=num_nodes_dict)
    train_pos_g.edges['similar'].data['sim'] = edge_weight2
    train_pos_g.edges['similar-by'].data['sim'] = edge_weight2

    # sub Graph: train_neg_g 
    train_neg_g = dgl.heterograph({
    ('drug_id', 'relate', 'side_id'): (train_neg_u, train_neg_v),
    ('side_id', 'relate-by', 'drug_id'): (train_neg_v, train_neg_u),
    ('drug_id','similar','drug_id'):(src,dst),
    ('drug_id','similar-by','drug_id'):(dst,src)

    },num_nodes_dict=num_nodes_dict)
    train_neg_g.edges['similar'].data['sim'] = edge_weight2
    train_neg_g.edges['similar-by'].data['sim'] = edge_weight2
    test_pos_g = dgl.heterograph({
    ('drug_id', 'relate', 'side_id'): (test_pos_u, test_pos_v),
    ('side_id', 'relate-by', 'drug_id'): (test_pos_v, test_pos_u),
    ('drug_id','similar','drug_id'):(src,dst),
    ('drug_id','similar-by','drug_id'):(dst,src)

    },num_nodes_dict=num_nodes_dict)
    test_pos_g.edges['similar'].data['sim'] = edge_weight2
    test_pos_g.edges['similar-by'].data['sim'] = edge_weight2

    # sub Graph: test_neg_g
    test_neg_g = dgl.heterograph({
    ('drug_id', 'relate', 'side_id'): (test_neg_u, test_neg_v),
    ('side_id', 'relate-by', 'drug_id'): (test_neg_v, test_neg_u),
    ('drug_id','similar','drug_id'):(src,dst),
    ('drug_id','similar-by','drug_id'):(dst,src)

    },num_nodes_dict=num_nodes_dict)
    test_neg_g.edges['similar'].data['sim'] = edge_weight2
    test_neg_g.edges['similar-by'].data['sim'] = edge_weight2 

    # train_g: remove test set only
    train_g = dgl.remove_edges(G, eids_pm[offset:offset+fsz],'relate') 

    train_g = dgl.remove_edges(train_g, eids_pm[offset:offset+fsz],'relate-by') 

    model = Model(10, 20, 5, train_g.etypes) # inputfeatures=10, hidden_features=20, output_features=1, relation_names: train_g.etypes
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    for epoch in range(300):
        pos_score, neg_score = model(train_g,train_pos_g, train_neg_g, ('drug_id', 'relate', 'side_id'))
        loss = compute_loss(pos_score, neg_score)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if epoch % 20==0:
            print('the',epoch,'loss=',loss.item())
            with torch.no_grad(): 
                roc_auc_score_train, pr_auc_score_train , f1_micro_train, _ ,  _ ,  _ ,  _ ,  _ ,  _ , _ ,  _,                prec, recall, fpr, tpr= compute_auc(pos_score, neg_score)
                # print('roc_auc:',roc_auc_score_train,'f1 micro:',f1_micro_train)

    # test set
    with torch.no_grad():
        print("Result:")
        print('AUC-ROC-TRAIN:', roc_auc_score_train,'AUC-PR-TRAIN:', pr_auc_score_train)
        AUC_roc_train[f] = roc_auc_score_train
        AUC_pr_train[f] = pr_auc_score_train
        pos_score, neg_score = model(train_g,test_pos_g, test_neg_g, ('drug_id', 'relate', 'side_id'))
        roc_auc_score_test,pr_auc_score_test, f1_micro_test, f1_macro_test, f1_weight_test, prec_micro_test,        prec_macro_test, prec_weight_test, recall_micro_test,recall_macro_test, recall_weight_test        ,prec, recall, fpr, tpr = compute_auc(pos_score, neg_score)
        print('AUC-ROC-TEST:', roc_auc_score_test,'AUC-PR-TEST:', pr_auc_score_test)
        AUC_roc_test[f] = roc_auc_score_test
        AUC_pr_test[f] = pr_auc_score_test
        F1_micro_test[f] = f1_micro_test
        F1_macro_test[f]  = f1_macro_test
        F1_weight_test[f]  = f1_weight_test
        Prec_micro_test[f]  = prec_micro_test
        Prec_macro_test[f]  = prec_macro_test
        Prec_weight_test[f]   = prec_weight_test
        Recall_micro_test[f]  = recall_micro_test
        Recall_macro_test[f]  = recall_macro_test
        Recall_weight_test[f] = recall_weight_test

        algod['fold'+str(f)] = {}
        algod['fold'+str(f)]['gcnmlp'] = {}
        algod['fold'+str(f)]['gcnmlp']['fpr'] = fpr
        algod['fold'+str(f)]['gcnmlp']['tpr'] = tpr
        algod['fold'+str(f)]['gcnmlp']['roc_auc'] = roc_auc_score_test
        algod['fold'+str(f)]['gcnmlp']['prec'] = prec
        algod['fold'+str(f)]['gcnmlp']['recall'] = recall
        algod['fold'+str(f)]['gcnmlp']['auc_pr'] = pr_auc_score_test
        algod['fold'+str(f)]['gcnmlp']['F1_macro'] = f1_macro_test
        algod['fold'+str(f)]['gcnmlp']['Prec_macro'] = prec_macro_test
        algod['fold'+str(f)]['gcnmlp']['Recall_macro'] = recall_macro_test
    offset += fsz
print("Mean +/- SD")
print("Mean AUC ROC TEST", AUC_roc_test.mean()," ", "SD:", AUC_roc_train.std())
print("Mean AUC PR TEST", AUC_pr_test.mean()," ", "SD:", AUC_pr_train.std())



 


 

 

 
