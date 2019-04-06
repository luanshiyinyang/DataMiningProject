# 员工离职预测
- 简介
	- DC的一道回归预测题。
	- 比较基础的分类，主要对逻辑回归的使用。
	- 核心思路为**属性构造+逻辑回归**。
- 过程
	- 数据获取
		- 报名参与比赛即可获得数据集的百度网盘地址，这个比赛时间很久，随时可以报名。
	- 数据探索
		- 无关项
			- EmployeeNumber为编号，对建模是干扰项，删除即可。
			- StandardHours和Over18全数据集固定值，没有意义，删除。
		- 相关性高
			- 相关图
				- ![](https://img-blog.csdnimg.cn/20190406125256512.png)
			- 可以发现，有两项相关性极高，删除其中一个MonthlyIncome。
	- 数据预处理
		- one-hot编码
			- 对几个固定几个取字符串值的特征进行one-hot编码
		- 属性构造
			- 特征数目较少，暴力拼接不同属性，构造新属性
	- 数据挖掘建模
		- 既是回归赛又是分类题，很明显就是使用逻辑回归（LR）模型。
		- 但是还是使用未调参的几个基础模型进行交叉验证，发现LR较高，加上其他模型调参麻烦，就没有多加研究。
			- 代码
				- ```python
					# 多模型交叉验证
					from sklearn.linear_model import LogisticRegression
					from sklearn.svm import SVC
					from sklearn.tree import DecisionTreeClassifier
					from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
					import sklearn.neural_network as sk_nn
					from sklearn.model_selection import cross_val_score
					models = {
					    'LR': LogisticRegression(solver='liblinear', penalty='l2', C=1),
					    'SVM': SVC(C=1, gamma='auto'),
					    'DT': DecisionTreeClassifier(),
					    'RF' : RandomForestClassifier(n_estimators=100),
					    'AdaBoost': AdaBoostClassifier(n_estimators=100),
					    'GBDT': GradientBoostingClassifier(n_estimators=100),
					    'NN': sk_nn.MLPClassifier(activation='relu',solver='adam',alpha=0.0001,learning_rate='adaptive',learning_rate_init=0.001, max_iter=1000)  
					}
					
					for k, clf in models.items():
					    print("the model is {}".format(k))
					    scores = cross_val_score(clf, x_train, y_train, cv=10)
					    print(scores)
					    print("Mean accuracy is {}".format(np.mean(scores)))
					    print("*" * 100)
					```
		- 对LR进行网格搜索调参，发现默认参数即可有不错的平台验证率。
			- 代码
				- ```python
					# 网格搜索调参
					from sklearn.model_selection import GridSearchCV
					from sklearn.linear_model import LogisticRegression
					penaltys = ['l1', 'l2']
					Cs = np.arange(1, 10, 0.1)
					parameters = dict(penalty=penaltys, C=Cs )
					lr_penalty= LogisticRegression(solver='liblinear')
					grid= GridSearchCV(lr_penalty, parameters,cv=10)
					grid.fit(x_train,y_train)
					grid.cv_results_
					print(grid.best_score_)
					print(grid.best_params_) 
					```


- 补充说明
	- 其实XgBoost和RF可能效果更好一些，但是由于一些原因，没有深究，有兴趣的可以进一步研究，最高的貌似研究有0.92以上通过率了。
	- 具体数据集和代码可以在我的Github找到，result.csv即为提交文件。
	- 附上提交时的平台分数和排名（22/1808）。
		- ![](https://img-blog.csdnimg.cn/20190406125338529.png)