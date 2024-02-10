from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
import pandas as pd
import numpy as np
cell_df= pd.read_csv(r"F:\M.tech\2nd sem\data\cancer.csv")
cell_df = cell_df[pd.to_numeric(cell_df['BareNuc'], errors='coerce').notnull()]
cell_df['BareNuc'] = cell_df['BareNuc'].astype('int')
feature_df = cell_df[['Clump', 'UnifSize', 'UnifShape', 'MargAdh', 'SingEpiSize', 'BareNuc', 'BlandChrom', 'NormNucl', 'Mit']]
X = np.asarray(feature_df)
y = np.asarray(cell_df['Class'])
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
model = GaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = metrics.accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print("Confusion Matrix:")
c=metrics.confusion_matrix(y_test, y_pred, labels=[2,4])
print(c)
import seaborn as sns
import matplotlib.pyplot as plt
classes=['Benign(2)','Malignant(4)']
sns.heatmap(c,cmap="Blues",annot=True,fmt="d",xticklabels=classes,yticklabels=classes)
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

xnew=[[9,6,8,8,10,7,6,3,4],
      [3,2,1,5,6,8,10,2,1],
      [1,1,3,2,3,2,3,1,2],
      [8,7,5,10,7,9,5,5,4]]
ypred_new=model.predict(xnew)
print("Predicted for new entry\n",ypred_new)
for i in range (len(ypred_new)):
      if ypred_new[i]==2:
            print(f'sample {i} is Benign')
      else:
            print(f'sample {i} is Malignant')