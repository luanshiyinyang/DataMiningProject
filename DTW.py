# -*- coding:utf-8-*-

def DTW(s1, s2):
    """
    计算s1和s2两个向量之间的距离
    使用曼哈顿距离，如果时间序列输入是二维的那么建议使用欧几里得距离
    """
    import numpy as np
    from numpy import array, zeros, argmin, inf, equal, ndim
    from sklearn.metrics.pairwise import manhattan_distances
    r, c = len(s1), len(s2)
    D0 = zeros((r+1,c+1))
    D0[0,1:] = inf
    D0[1:,0] = inf
    D1 = D0[1:,1:]
    
    for i in range(r):
        for j in range(c):
            D1[i,j] = manhattan_distances(np.array(s1[i]).reshape(-1,1),np.array(s2[j]).reshape(-1,1))
    M = D1.copy()
    for i in range(r):
        for j in range(c):
            D1[i,j] += min(D0[i,j],D0[i,j+1],D0[i+1,j])
    i,j = array(D0.shape) - 2
    
    p,q = [i],[j]
    while(i>0 or j>0):
        tb = argmin((D0[i,j],D0[i,j+1],D0[i+1,j]))
        if tb==0 :
            i-=1
            j-=1
        elif tb==1 :
            i-=1
        else:
            j-=1
        p.insert(0,i)
        q.insert(0,j)
        return D1[-1,-1]
if __name__ == "__main__":
    # 测试DTW计算结果
    import numpy as np
    s1 = np.array([1, 2, 3, 4, 5, 5, 5, 4])
    s2 = np.array([3, 4, 5, 5, 5, 4])
    print(DTW(s1,s2))