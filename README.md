# 北京PM2.5浓度回归分析训练赛
- 简介
	- DC上的一个回归题，比较简单。
	- 时间原因没有细看，提交到70多名就结束了。
	- 使用stacking方法结合多个回归模型。
- 过程
	- 数据获取
		- 官方给定。
	- 数据探索
		- 训练集有35746条记录，13个字段，有表头，其中pm2.5为目标。
		- 叙述
			- ![](https://img-blog.csdnimg.cn/2019041220534547.png)
	- 数据预处理
		- 主要对date属性进行预处理，因为其字符串属性无法参与建模。
			- 利用time模块解析日期并生成新特征为年、月、日、周。
		- 还可以进行一些特征组合，时间关系，我就直接强代入模型了。
	- 数据挖掘建模
		- 平时比较喜欢将一个模型调参到合适，这次由于数据原因选择了stacking构建模型，使用mlxtend库。
		- 核心代码
			- ```python
				from sklearn.linear_model import LinearRegression, Ridge, Lasso
				from sklearn.tree import DecisionTreeRegressor
				from sklearn.svm import SVR
				from sklearn.neighbors import KNeighborsRegressor
				lr = LinearRegression()
				dtr = DecisionTreeRegressor()
				svr_rbf = SVR(kernel='rbf', gamma='auto')
				knr = KNeighborsRegressor()
				ridge = Ridge()
				lasso = Lasso()
				regression_models = [lr, dtr, svr_rbf, knr, ridge, lasso]
				from mlxtend.regressor import StackingCVRegressor
				sclf = StackingRegressor(regression_models, meta_regressor=ridge)
				sclf.fit(x_tra, y_tra)
				```
			- **mlxtend的模型是可以使用sklearn库进行网格搜索调参的。**
		- 验证集拟合情况
			- ![](https://img-blog.csdnimg.cn/20190412210207765.png)
- 补充说明
	- 如果继续调参会有不错的分数。
	- 数据集和代码见我的Github，欢迎star或者fork。
	- 附上提交时的排名（76/832)。
		- ![](https://img-blog.csdnimg.cn/20190412210353516.png)