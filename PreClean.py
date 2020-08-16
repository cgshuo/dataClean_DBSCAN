import datetime
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

#-----主程序-----------------------------------------------------------------------------------------------
#----------------------------------------------数据预处理---------------------------------------------------
print("----------------------------------------------数据预处理---------------------------------------------------")
print("--------------------------LoadFile-------------------------------------")
print("------开始读取文件------",datetime.datetime.now())
path='data/2015.csv'
with open(path, encoding='gbk') as file:
    data=pd.read_csv(file)
    #print(data) #初始数据：13994rows * 19columns
df=pd.DataFrame(data)
print("原始数据：[13994 rows x 19 columns]")

#-----------------去除电为0的记录----------------------------------
print("-----------------Filter data with power capacity of 0------------------")
index_0=df[df.电==0]['电'].index.tolist()
print("电消耗为0样本点索引：",index_0) #电为0的数据(共20rows*19colunms)
print(df[df.电==0])
df.drop(index=index_0,inplace=True)
print("------Complete-------",datetime.datetime.now())
print(df)
print("电为0的样本点：电为0的样本点： Emmpty DataFrame")
print("原始数据：[13994 rows x 19 columns], 13994个样本点")
print("现存数据：[13974 rows x 19 columns]")
#---------------------------------------------------------------------------------------------------

print("-----------------——————————————-----------去重-----------————————————————--------------------")
mainData=df.loc[:, ['建筑名称', '竣工年度', '建筑面积', '电', '煤炭', '天然气', '液化石油气', '人工煤气']]
countDup = 0
dupIndex=[]
Indexj=-1;
for i in mainData.duplicated():
    Indexj=Indexj+1;
    if i == True:
        countDup = countDup + 1
        dupIndex.append(Indexj)
print("现存数据：[13974 rows x 19 columns], 13974个样本点")
dflength=len(df.index)
dupRate=countDup/dflength
print("重复样本点个数", countDup,"|","重复率",'%.2f%%' % (dupRate * 100))
dupIndex2=[2475, 2476,2479, 2480,2482, 2483,2485, 2486,2489, 2490,2491, 2492,2493, 2494,2495, 2496,2497, 2498]
pd.set_option('display.unicode.ambiguous_as_wide', True) #jr中文对齐
pd.set_option('display.unicode.east_asian_width', True)
pd.set_option('display.max_columns', None)
pd.set_option('display.width',1000)
print("重复样本索引", dupIndex2)
print(mainData.iloc[dupIndex2])
df.drop(index=dupIndex,inplace=True)
print("------Complete-------",datetime.datetime.now())
print(df)
print("重复样本点个数 0","|","重复率 0.0%")
print("原始数据：[13994 rows x 19 columns], 13994个样本点")
print("现存数据：[13965 rows x 19 columns], 13965个样本点")
#---------------------------------------------------------------------------------------------------
#----------------------------------计算平均能耗-----------------------------------------------------------------
df['单位面积耗电量']= df['电']/df['建筑面积']
df['单位面积耗煤炭量']= df['煤炭']/df['建筑面积']
df['单位面积耗天然气量']= df['天然气']/df['建筑面积']
df['单位面积耗液化石油气量']= df['液化石油气']/df['建筑面积']
df['单位面积耗人工煤气量']= df['人工煤气']/df['建筑面积']

#---------------------------------------------------------------------------------------------------

#-----------------------------------------箱线图----------------------------------------------------------
print("#-----------------------------------------箱线图----------------------------------------------------------")
BoxData = df.loc[:,['单位面积耗电量']]
# 计算上下四分位数
Q1 = BoxData.单位面积耗电量.quantile(q = 0.25)
Q3 = BoxData.单位面积耗电量.quantile(q = 0.75)
print("计算上下四分位数:Q1 = BoxData.单位面积耗电量.quantile(q = 0.25)=",Q1)
print("计算上下四分位数:Q3 = BoxData.单位面积耗电量.quantile(q = 0.75)=",Q3)
#异常值判断标准， 1.5倍的四分位差 计算上下须对应的值
low_quantile = Q1 - 1.5*(Q3-Q1)
high_quantile = Q3 + 1.5*(Q3-Q1)
print("计算异常值判断标准， 1.5倍的四分位差，计算上下须对应的值：")
print("low_quantile = Q1 - 1.5*(Q3-Q1)",low_quantile)
print("high_quantile = Q3 + 1.5*(Q3-Q1)",high_quantile)

# 输出异常值
value = BoxData.单位面积耗电量[(BoxData.单位面积耗电量 > high_quantile) | (BoxData.单位面积耗电量 < low_quantile)]
index_box=value.index.tolist()
df.drop(index=index_box,inplace=True)
print("异常样本个数",len(index_box))
print(value)
print("------Complete-------",datetime.datetime.now())
print(df)
print("原始数据：[13994 rows x 19 columns], 13994个样本点")
print("现存数据：[12836 rows x 24 columns], 12836个样本点")

# #绘制箱线图
plt.rcParams['font.sans-serif'] = ['Songti SC']
plt.rcParams['axes.unicode_minus'] = False
f = BoxData.boxplot(
               sym = 'o',            #异常点形状
               vert = True,          # 是否垂直
               whis=1.5,             # IQR
               patch_artist = True,  # 上下四分位框是否填充
               meanline = False,showmeans = True,  # 是否有均值线及其形状
               showbox = True,   # 是否显示箱线
               showfliers = True,  #是否显示异常值
               notch = False,    # 中间箱体是否缺口
               return_type='dict')  # 返回类型为字典
plt.show()
#---------------------------------------------------------------------------------------------------

#-----------------------------------------聚类-------------------------------------------------------
print("-----------------------------------------聚类----------------------------------------------------------")
print("------抽取数据------",datetime.datetime.now())
Dbsacan_data = df.loc[:, ['单位面积耗电量', '单位面积耗煤炭量', '单位面积耗天然气量', '单位面积耗液化石油气量', '单位面积耗人工煤气量']]
#Dbsacan_data = df.loc[:, [ '电', '煤炭', '天然气', '液化石油气', '人工煤气']]
print("------开始聚类------",datetime.datetime.now())
eps=4 #75
min_samples=8 #3
db = DBSCAN(eps, min_samples).fit(Dbsacan_data)
labels = db.labels_
Dbsacan_data['cluster_db'] = labels  # 在数据集最后一列加上经过DBSCAN聚类后的结果(噪声为-1）
Dbsacan_data.sort_values('cluster_db')
print("------聚类完成，输出结果------",datetime.datetime.now())

# pd.set_option('display.max_columns', None)
# pd.set_option('display.width',1000)
#查看根据DBSCAN聚类后的分组统计结果（均值)
print("查看根据DBSCAN聚类后的分组统计结果（均值）")
print(Dbsacan_data.groupby('cluster_db').mean())

# print("------计算轮廓系数------",datetime.datetime.now())
# score = metrics.silhouette_score(Dbsacan_data,Dbsacan_data.cluster_db)
# print("eps",eps,"minpts",min_samples,"轮廓系数",score) #接近1就好

index_dbscan=Dbsacan_data[Dbsacan_data.cluster_db==-1].index.tolist()
print("异常样本个数",len(index_dbscan))
print(Dbsacan_data[Dbsacan_data['cluster_db']==-1])

print("------根据聚类结果清洗异常点------",datetime.datetime.now())
df.drop(index=index_dbscan,inplace=True)

print("------Complete-------",datetime.datetime.now())
print(df)
print("原始数据：[13994 rows x 19 columns], 13994个样本点")
print("现存数据：[12698 rows x 24 columns], 12698个样本点")

print("-----绘图------",datetime.datetime.now())
plt.rcParams['font.sans-serif'] = ['Songti SC']
plt.rcParams['axes.unicode_minus'] = False
pd.plotting.scatter_matrix(Dbsacan_data, marker='o',c=Dbsacan_data.cluster_db, figsize=(10,10), s=20)
plt.show()
#---------------------------------------------------------------------------------------------------
