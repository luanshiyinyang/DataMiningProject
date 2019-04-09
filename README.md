# 美国King County房价预测训练赛
- 简介
	- DC上的一个回归题（正经的回归题）。
	- 比较简单。
	- 时间原因（暂时没什么时间看国内旧赛），看了一下网上的解答，改善了一下神经网络就提交了。
- 过程
	- 数据获取
		- 报名成功后到官网提供的入口下载，或者我的Github也上传了。
	- 数据探索
		- 简单了解数据格式。
			- 训练集有10000条记录，14个特征，描述如下。（注意，官方数据集没有表头）
				- ![](https://img-blog.csdnimg.cn/20190409210414105.png)
				- 其中，第二列“销售价格”就是目标。
			- 测试集有3000条记录，利用训练好的模型预测这3000条记录的房价。
	- 数据预处理
		- 设置表头
			- 原数据没有表头，自己补充即可。
		- 显然，实际数据销售日期是有意义的，但是，对模型建立不方便，提取年份，删除月日。
		- 利用销售日期组合修理及建造日期构建新特征。
		- 处理后数据集落地。
	- 数据挖掘建模
		- 几种回归尝试
			- 随机森林（RFR）
			- 线性回归
		- 神经网络
			- 由于几种回归表现一般，没有再尝试，看网上分享很多神经网络做法，参考设计了一个前馈网络。
			- 使用Keras（TensorFlow作为后端，GPU训练）
			- ![](https://img-blog.csdnimg.cn/20190409212758337.png)
			- ![](https://img-blog.csdnimg.cn/20190409212901962.png)
			- 训练5000次左右提交为100名成绩。
			- 注意：**5000次之前就已经收敛，为了效率可以加入EarlyStopping。（时间原因，没有处理）
			- ![](https://img-blog.csdnimg.cn/2019040921311813.png)
		- 网络代码
			- ```python
				model = Sequential()
				input_size = len(df_train.columns)
				model.add(Dense(units=90, activation='relu', input_shape=(input_size, )))
				model.add(Dropout(0.5))
				model.add(Dense(units=45, activation='relu'))
				model.add(Dropout(0.5))
				model.add(Dense(units=30,activation='relu'))
				model.add(Dropout(0.25))
				model.add(Dense(units=15, activation='relu'))
				model.add(Dropout(0.1))
				# 此处不能使用激活函数，因为放假是放射的
				model.add(Dense(units=1,activation=None))
				# 官网使用mse计算损失
				model.compile(loss='mean_squared_error',optimizer='adam',metrics=[metrics.mae])
				model.summary()
				```
- 补充说明
	- 排名靠前的应该不少使用机器学习算法回归调参，有时间的不妨一试。
	- 具体数据集和代码见我的Github，欢迎Star或者Fork（环境为Jupyter）。
	- 附上提交时的排名。
		- ![](https://img-blog.csdnimg.cn/20190409213257720.png)