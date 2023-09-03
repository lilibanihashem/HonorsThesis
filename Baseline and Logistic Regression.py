#!/usr/bin/env python
# coding: utf-8

# In[37]:


import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import plotly.express as px


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score
from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[38]:


#Reading in data

data = pd.read_csv("Cleaning_Output")
train_X, train_Y = pd.read_csv("train_X_CUT"), pd.read_csv("train_Y_CUT")["WF9_CESD7_CUT"]
dev_X, dev_Y = pd.read_csv("dev_X_CUT"), pd.read_csv("dev_Y_CUT")["WF9_CESD7_CUT"]
test_X, test_Y = pd.read_csv("test_X_CUT"), pd.read_csv("test_Y_CUT")["WF9_CESD7_CUT"]


# In[39]:


train_X.shape


# ### Metrics
# 
# 1. TPR
# 2. FPR
# 3. Precision
# 4. Accuracy

# In[40]:


def showmetrics(pred, true):
    cm = confusion_matrix(true, pred)
    acc = (cm.ravel()[0]+cm.ravel()[3])/sum(cm.ravel())
    TPR = cm.ravel()[3]/(cm.ravel()[3]+cm.ravel()[2])
    FPR = cm.ravel()[1]/(cm.ravel()[1]+cm.ravel()[0])
    prec = cm.ravel()[3]/(cm.ravel()[3]+cm.ravel()[1])
    f1 = 2*((prec*TPR)/(prec+TPR))

    print(cm)
    print("Model TPR: " + str(TPR))
    print("Model FPR: " + str(FPR))
    print("Model F1: " + str(f1))
    print("Model Precision: " + str(prec))
    print("Model Accuracy: " + str(acc))
    return acc, TPR, FPR, f1, prec


# ## Baseline

# In[41]:


model = LogisticRegression()
model.fit(train_X, train_Y)

pred_base = [0 for x in model.predict(test_X)]

showmetrics(pred_base, test_Y)


# ## Training Model

# In[42]:


train_X = sm.add_constant(train_X)
dev_X = sm.add_constant(dev_X)


# In[43]:


model = sm.Logit(train_Y, train_X).fit()


# In[44]:


model.summary()


# In[45]:


pred = [1 if x == True else 0 for x in model.predict(dev_X) > 0.5]


# In[46]:


acc1, TPR1, FPR1, f1_1, prec1 = showmetrics(pred, dev_Y)


# # VIF

# In[47]:


vif_drop = train_X


# In[48]:


# VIF dataframe
vif_data = pd.DataFrame()
vif_data["feature"] = vif_drop.columns


# In[49]:


vif_data["VIF"] = [variance_inflation_factor(vif_drop.values, i)
                          for i in range(len(vif_drop.columns))]
vif_data.head(60)


# ## ROC Curve

# In[50]:


pred_prob = model.predict(dev_X)

fpr, tpr, proba = roc_curve(dev_Y, pred_prob)

plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(dev_Y, pred_prob)}")
plt.title("ROC Curve")
plt.ylabel("True Positive Rate")
plt.xlabel("False Positive Rate")
plt.legend(loc="lower right")


# ## Predictions With "Optimal" Threshold

# In[51]:


optimal_proba_cutoff = sorted(list(zip(np.abs(tpr - fpr), proba)), key=lambda i: i[0], reverse=True)[0][1]
optimal_proba_cutoff


# In[52]:


pred = [1 if x == True else 0 for x in model.predict(dev_X) > 0.052669540097475515]


# In[53]:


acc2, TPR2, FPR2, f1_2, prec2 = showmetrics(pred, dev_Y)


# ## Predictions with Threshold = 0.2

# In[54]:


pred = [1 if x == True else 0 for x in model.predict(dev_X) > 0.2]


# In[55]:


acc3, TPR3, FPR3,f1_3, prec3 = showmetrics(pred, dev_Y)


# # Predictions on Test Data

# In[56]:


test_X, test_Y = pd.read_csv("test_X_CUT"), pd.read_csv("test_Y_CUT")["WF9_CESD7_CUT"]
test_X = sm.add_constant(test_X)


# In[57]:


pred = [1 if x == True else 0 for x in model.predict(test_X) > 0.5]


# In[58]:


acc_t_1, TPR_t_1, FPR_t_1 ,f1_t_1, prec_t_1 = showmetrics(pred, test_Y)


# In[ ]:





# In[59]:


pred = [1 if x == True else 0 for x in model.predict(test_X) > 0.1]


# In[60]:


acc_t_2, TPR_t_2, FPR_t_2 ,f1_t_2, prec_t_2 = showmetrics(pred, test_Y)


# In[ ]:





# In[61]:


pred_prob = model.predict(test_X)

fpr, tpr, proba = roc_curve(test_Y, pred_prob)

fig = px.line(x=fpr, y=tpr, title=f"AUC = {roc_auc_score(test_Y, pred_prob)}")
fig.update_layout(xaxis_title = "False Positive Rate", yaxis_title ="True Positive Rate", title = "ROC Curve", width = 1000, 
                  height = 800)
fig.add_annotation(x=0.77, y=0.016,
            text=f"AUC = {roc_auc_score(test_Y, pred_prob)}",
                               showarrow=False,
    font=dict(
            size=18,
            )
        )

fig.show()


# In[62]:


optimal_proba_cutoff = sorted(list(zip(np.abs(tpr - fpr), proba)), key=lambda i: i[0], reverse=True)[0][1]
optimal_proba_cutoff


# In[63]:


pred = [1 if x == True else 0 for x in model.predict(test_X) > 0.06407859132153843]


# In[64]:


acc_t_3, TPR_t_3, FPR_t_3 ,f1_t_3, prec_t_3 = showmetrics(pred, test_Y)


# In[ ]:





# In[65]:


pred = [1 if x == True else 0 for x in model.predict(test_X) > 0.08]


# In[66]:


acc_t_4, TPR_t_4, FPR_t_4 ,f1_t_4, prec_t_4 = showmetrics(pred, test_Y)


# In[67]:


pred = [1 if x == True else 0 for x in model.predict(test_X) > 0.2]


# In[68]:


acc_t_5, TPR_t_5, FPR_t_5 ,f1_t_5, prec_t_5 = showmetrics(pred, test_Y)


# ## Summary Table

# In[69]:


index = ["F1", "TPR", "FPR",  "Precision", "Accuracy"]
columns = ["Baseline", "T = 0.5", "T = 0.2","T = 0.1", "T = 0.08", "T = 0.064"]
tpr = [0, TPR_t_1, TPR_t_5, TPR_t_2, TPR_t_4, TPR_t_3]
fpr = [0, FPR_t_1, TPR_t_5, FPR_t_2, FPR_t_4, FPR_t_3]
f1 = ["n/a", f1_t_1, f1_t_5, f1_t_2, f1_t_4, f1_t_3]
prec = ["n/a", prec_t_1, prec_t_5, prec_t_2, prec_t_4, prec_t_3]
acc = [.93, acc_t_1, acc_t_5, acc_t_2, acc_t_4, acc_t_3]
rows = [f1, tpr, fpr, prec, acc]


# In[70]:


results = pd.DataFrame(rows, index = index, columns = columns)


# In[71]:


results


# In[72]:


results.to_csv("LogisticRegressionResults.csv")


# In[ ]:




