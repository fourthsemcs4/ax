from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
iris=load_iris()
loo=LeaveOneOut()
scores=[]
for train_index,test_index in loo.split(iris.data):
    x_train,x_test=iris.data[train_index],iris.data[test_index]
    y_train,y_test=iris.target[train_index],iris.target[test_index]
clf=LogisticRegression().fit(x_train,y_train)
y_pred=clf.predict(x_test)
scores.append(accuracy_score(y_test,y_pred))
scores=sum(scores)/len(scores)
print(f'accuracy:{scores:.2f}')