import datetime
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics

#-----主程序-----------------------------------------------------------------------------------------------
#----------------------------------------------数据预处理---------------------------------------------------
from matplotlib import colors
from sklearn.cluster import DBSCAN

print("----------------------------------------------数据预处理---------------------------------------------------")
print("--------------------------LoadFile-------------------------------------")

print("------开始读取文件------",datetime.datetime.now())
path='2015.csv'
with open(path, encoding='gbk') as file:
    data=pd.read_csv(file)
    #print(data) #初始数据：13994rows * 19columns
print("原始数据：[13994 rows x 19 columns]")

#-----------------Filter data with power capacity of 0------------------
print("-----------------Filter data with power capacity of 0------------------")

df=pd.DataFrame(data)
index_0=df[df.电==0]['电'].index.tolist()
print("电消耗为0样本点索引：",index_0) #电为0的数据(共20rows*19colunms)
df.drop(index=index_0,inplace=True)
print("------Complete-------",datetime.datetime.now())
print(df)
print("电为0的样本点：",df[df.电==0])
print("现存数据：[13974 rows x 19 columns]")

#---------------------------------------------------------------------------------------------------
print("----------------------------Deduplication-------------------------------")

mainData=df.loc[:, ['建筑编码', '竣工年度', '建筑面积', '电', '煤炭', '天然气', '液化石油气', '人工煤气']]
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
pd.set_option('display.unicode.ambiguous_as_wide', True) #中文对齐
pd.set_option('display.unicode.east_asian_width', True)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.width',1000)
print("重复样本索引", dupIndex2)
print(mainData.iloc[dupIndex2])
df.drop(index=dupIndex,inplace=True)
print("------Complete-------",datetime.datetime.now())
print(df)
print("重复样本点个数 0","|","重复率 0.0%")
print("原始数据：[13994 rows x 19 columns], 13994个样本点")
print("现存数据：[13965 rows x 19 columns], 13965个样本点")

#---------------------------------------------------------------------------------------------------

df['平均耗电量']= df['电']/df['建筑面积']
df['平均耗煤炭量']= df['煤炭']/df['建筑面积']
df['平均耗天然气量']= df['天然气']/df['建筑面积']
df['平均耗液化石油气量']= df['液化石油气']/df['建筑面积']
df['平均耗人工煤气量']= df['人工煤气']/df['建筑面积']
# BoxData = df.loc[:, ['电单位面积', '煤单位面积', '天然气单位面积', '液化石油气单位面积', '人工煤气单位面积']]
BoxData = df.loc[:,['平均耗电量']]

# 计算上下四分位数
Q1 = BoxData.平均耗电量.quantile(q = 0.25)
Q3 = BoxData.平均耗电量.quantile(q = 0.75)
#异常值判断标准， 1.5倍的四分位差 计算上下须对应的值
low_quantile = Q1 - 1.5*(Q3-Q1)
high_quantile = Q3 + 1.5*(Q3-Q1)

# 输出异常值 电1132
value = BoxData.平均耗电量[(BoxData.平均耗电量 > high_quantile) | (BoxData.平均耗电量 < low_quantile)]
index_box=value.index.tolist()
#df.drop(index=index_box,inplace=True)
print("异常样本个数",len(index_box))
# # #绘制箱线图
# plt.rcParams['font.sans-serif'] = ['Songti SC']
# plt.rcParams['axes.unicode_minus'] = False
# f = BoxData.boxplot(
#                sym = 'o',            #异常点形状
#                vert = True,          # 是否垂直
#                whis=1.5,             # IQR
#                patch_artist = True,  # 上下四分位框是否填充
#                meanline = False,showmeans = True,  # 是否有均值线及其形状
#                showbox = True,   # 是否显示箱线
#                showfliers = True,  #是否显示异常值
#                notch = False,    # 中间箱体是否缺口
#                return_type='dict')  # 返回类型为字典
plt.show()

#---------------------------------------------------------------------------------------------------

BoxData = df.loc[:,['平均耗煤炭量']]

# 计算上下四分位数
Q1 = BoxData.平均耗煤炭量.quantile(q = 0.25)
Q3 = BoxData.平均耗煤炭量.quantile(q = 0.75)
#异常值判断标准， 1.5倍的四分位差 计算上下须对应的值
low_quantile = Q1 - 1.5*(Q3-Q1)
high_quantile = Q3 + 1.5*(Q3-Q1)

# 输出异常值 电1132
value = BoxData.平均耗煤炭量[(BoxData.平均耗煤炭量 > high_quantile) | (BoxData.平均耗煤炭量 < low_quantile)]
index_box=value.index.tolist()
#df.drop(index=index_box,inplace=True)
print("异常样本个数",len(index_box))
#---------------------------------------------------------------------------------------------------

BoxData = df.loc[:,['平均耗天然气量']]

# 计算上下四分位数
Q1 = BoxData.平均耗天然气量.quantile(q = 0.25)
Q3 = BoxData.平均耗天然气量.quantile(q = 0.75)
#异常值判断标准， 1.5倍的四分位差 计算上下须对应的值
low_quantile = Q1 - 1.5*(Q3-Q1)
high_quantile = Q3 + 1.5*(Q3-Q1)

# 输出异常值 电1132
value = BoxData.平均耗天然气量[(BoxData.平均耗天然气量 > high_quantile) | (BoxData.平均耗天然气量 < low_quantile)]
index_box=value.index.tolist()
#df.drop(index=index_box,inplace=True)
print("异常样本个数",len(index_box))
#---------------------------------------------------------------------------------------------------

BoxData = df.loc[:,['平均耗液化石油气量']]

# 计算上下四分位数
Q1 = BoxData.平均耗液化石油气量.quantile(q = 0.25)
Q3 = BoxData.平均耗液化石油气量.quantile(q = 0.75)
#异常值判断标准， 1.5倍的四分位差 计算上下须对应的值
low_quantile = Q1 - 1.5*(Q3-Q1)
high_quantile = Q3 + 1.5*(Q3-Q1)

# 输出异常值 电1132
value = BoxData.平均耗液化石油气量[(BoxData.平均耗液化石油气量 > high_quantile) | (BoxData.平均耗液化石油气量 < low_quantile)]
index_box=value.index.tolist()
#df.drop(index=index_box,inplace=True)
print("异常样本个数",len(index_box))
#---------------------------------------------------------------------------------------------------

BoxData = df.loc[:,['平均耗人工煤气量']]

# 计算上下四分位数
Q1 = BoxData.平均耗人工煤气量.quantile(q = 0.25)
Q3 = BoxData.平均耗人工煤气量.quantile(q = 0.75)
#异常值判断标准， 1.5倍的四分位差 计算上下须对应的值
low_quantile = Q1 - 1.5*(Q3-Q1)
high_quantile = Q3 + 1.5*(Q3-Q1)

# 输出异常值 电1132
value = BoxData.平均耗人工煤气量[(BoxData.平均耗人工煤气量 > high_quantile) | (BoxData.平均耗人工煤气量 < low_quantile)]
index_box=value.index.tolist()
#df.drop(index=index_box,inplace=True)
print("异常样本个数",len(index_box))
