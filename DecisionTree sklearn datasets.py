#!/usr/bin/env python
# coding: utf-8

# ### 1. 문제정의(목표설정)
# - 버섯의 데이터를 활용해서 버섯이 독/식용 분류를 해보자!
# - 결정트리 모델을 시각화 해보자!

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
#결정 트리 모델 임포트
from sklearn.tree import DecisionTreeClassifier


# In[2]:


#1. 데이터를 로드(index설정 하지 않아도 됨)
#2. 전체 컬럼, 행 숫자 파악해보기
#3. 결측치가 있는지 확인해보기


# ### 2.데이터 수집(데이터 로드) 

# In[3]:


#data = pd.read_csv("mushroom.csv",index_col="poisonous")
data = pd.read_csv("mushroom.csv")
data.info()
data


# In[4]:


data.describe()
#top:해당컬럼에 가장 많은 빈도수를 차지하는 데이터
#freq: top에서 나온 데이터의 개수


# In[5]:


#data.index.value_counts()
#value_counts()
#data.shape


# ### 3.데이터 전처리
# - 학습용 데이터라 전처리가 필요하지 않다

# ### 4.탐색적 데이터 분석
# - 데이터의 일부만 시각화 해보자!

# In[6]:


#차트나 그래프로 시각화 할 수 있는 라이브러리
import seaborn as sns


# In[7]:


sns.countplot(data=data,
             x="cap-shape",
              #hue : 해당 컬럼의 유니크값들이 들어가서 출력됨
              # 유니크값에 대한 각각의 bar, 범례가 자동적으로 표시
              hue="poisonous"
             );


# In[8]:


sns.countplot(data=data,
             x="cap-surface",
             hue="poisonous");


# ### 5. 모델링
# - 문제(X) 정답(y)

# In[9]:


#정답은 poisonous
#문제는 나머지 컬럼 전부
X = data.iloc[:,1:24]
y = data.loc[:,"poisonous"]
# X = data.loc[:,"cap-shape":] #cap-shape 컬럼부터 우측 끝까지

#X_train, X_test, y_train, y_test = train_test_split(X,y,
 #                       test_size = 0.3, # 30% 를 평가용으로 사용하겠습니다!
  #                      random_state = 0 #난수 사이클 seed 값 고정하는 역할
   #                     )


# In[10]:


#print(X_train.shape)
#print(X_test.shape)
#print(y_train.shape)
#print(y_test.shape)


# ###  레이블 인코딩
# - 숫자의 크고 작음에 따라서 특성이 작용한다
# - 회귀와 같이 연속된 숫자를 다루는 알고리즘에서는 1,2,3~ 커지는 숫자에 따라서 중요도로 인식 될 수 있다
# - 그래서 기계가 잘못 이해하고 애매한 결과가 나오게 될 수 있다

# In[11]:


#cap-shape 컬럼의 유니크 값들만 출력해보자!
X["cap-shape"].unique()


# In[12]:


#cap-shape 컬럼의 유니크값들과 개수를 확인해보자!
X["cap-shape"].value_counts()


# In[13]:


#레이블 인코딩 해보기!
#진행 전
X["cap-shape"]


# In[14]:


#진행 후
#맵핑 ({}) 딕셔너리 형태
X["cap-shape"].map({"x":0,"f":1,"k":2,"b":3,"s":4,"c":5})


# ### one hot encoding
# - 모든 데이터를 0과 1로 변환
# - 컴퓨터는 binary 로 모든 데이터를 처리하기 때문에 레이블 보다 원핫인코딩을 주로 사용

# In[15]:


X


# In[16]:


X["cap-shape"].unique()


# In[17]:


#유니크의 개수에 따라 컬럼이 증가한다
#get_dummies 새로운 컬럼 추가 (분신 컬럼)
X_one_hot = pd.get_dummies(X)
X_one_hot


# - 정답 데이터의 경우 기계가 따로 학습할 필요가 없기 때문에 인코딩이 필요치 않다
# -  train 데이터의 정답과 test 데이터의 정답이 같은지 비교만 해주는 개념

# ##### 학습과 정답 데이터로 분리 

# In[18]:


X_train, X_test, y_train, y_test = train_test_split(X_one_hot,y,
                             test_size = 0.3,
                                                    random_state=10
                                                   )


# In[19]:


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# ### 모델링
# - 결정트리 모델을 가지고 와 학습 및 예측 시키고 정확도를 측정하자!

# In[20]:


# 결정트리모델은 하이퍼파라미터를 조정하지 않으면 완전히 분류될 때까지 분기해 나간다.
# 모델 불러오기
tree_model = DecisionTreeClassifier()

#모델 학습
tree_model.fit(X_train, y_train)


# In[21]:


tree_model.score(X_test, y_test)


# In[22]:


tree_model.score(X_train, y_train)


# ### 트리모델 내부 시각화
# - 외부 라이브러리 graphviz 설치 및 활용 (트리모델 전용 시각화 라이브러리)

# In[23]:


get_ipython().system('pip install graphviz')


# In[24]:


from sklearn.tree import export_graphviz


# In[25]:


#학습 된 모델 넣기
#dot파일 : 트리모델의 그래프를 그리기 위해 텍스트로 저장된 파일 형식
#dot파일화를 해야 시각화 가능
export_graphviz(tree_model, out_file="data/tree.dot",
               #클래스(label)
                class_names=["독", "식용"],
                #원핫인코딩 된 데이터프레임의 컬럼명을 가져와서 특성이름으로 지정
                feature_names=X_one_hot.columns,
                #지니 불순도 출력(false는 미출력)
                impurity=True,
                #True : 클래스가 구분되도록 색을 칠해줌
                filled=True
               )


# In[26]:


import os
os.environ["PATH"]+=os.pathsep+'C:/Program Files/Graphviz/bin/'


# In[27]:


import graphviz
# with open 으로 이미 있는 파일을 불러온다.
# tree.dot 파일을 불러오면서 국제기준 인코딩 UTF-8로 변환시키고 f로 지칭
with open("data/tree.dot", encoding="UTF8") as f:
    dot_graph=f.read()
    
display(graphviz.Source(dot_graph))

# <= 0.5 : 원핫인코딩이 된 후 0또는 1을 구하는 조건식
# gini : 지니불순도 값
# value : 클래스별 개수 ["독", "식용"]
# class : 모델이 예측한 값


# ### 하이퍼 파라미터 조정하기

# In[28]:


tree_model2 = DecisionTreeClassifier(max_depth=3, max_leaf_nodes=5, min_samples_leaf=30)

tree_model2.fit(X_train, y_train)


# In[29]:


tree_model2.score(X_train, y_train)


# In[30]:


tree_model2.score(X_test, y_test)


# In[31]:


export_graphviz(tree_model2, out_file="data/tree2.dot",
                class_names=["독", "식용"],
                feature_names=X_one_hot.columns,
                impurity=True,
                filled=True
               )


# In[32]:


import graphviz
with open("data/tree2.dot", encoding="UTF8") as f:
    dot_grap2=f.read()
    
display(graphviz.Source(dot_grap2))


# ### 하이퍼 파라미터를 조정해가면서 그래프로 그려보자 

# In[33]:


train_list=[]
test_list=[]

for k in range(1,10):
    #모델 생성 및 하이퍼 파라미터에 변수 넣기
    tree_model = DecisionTreeClassifier(max_depth = k)
    #모델 학습
    tree_model.fit(X_train, y_train)
    #학습한 모델의 정확도 측정 -> 측정한 값은 리스트에 담기
    #학습데이터
    train_score = tree_model.score(X_train, y_train)
    train_list.append(train_score)
    
    #평가데이터
    tree_score = tree_model.score(X_test,y_test)
    test_list.append(tree_score)


# In[34]:


#max_depth 값 변화에 따른 train, test 데이터의 정확도 추이
plt.figure(figsize=(15,5))
plt.plot(train_list, label="train")
plt.plot(test_list, label="test")
#loc : 범례위치 (uppper, center, lower),(left, center, right)
plt.legend(loc="upper left", prop={"size":20})
plt.show()


# ###  특성선택(feature selection)
# - 지도학습 모델에서 데이터의 각 특성들의 중요도를 출력할 수 있음
# - 각 특성들은 0~1 사이의 중요도 값을 가짐, 모든 특성의 중요도 총합은 1이 됨
# - 0이나 1은 극단적인 특성이므로 도움이 되지 않는다.
# - 0 : 결과값에 아무런 영향을 안 주는 특성
# - 1 : 100% 영향을 주기에 굳이 ML이 필요하지 않음

# In[35]:


#트리모델의 각 특성들의 중요도 출력
fi = tree_model.feature_importances_
print(fi)


# In[36]:


fi_df = pd.DataFrame(fi, index=X_one_hot.columns, columns=["특성중요도"]) 
fi_df = fi_df * 100
fi_df.iloc[:,0]


# In[37]:


pd.options.display.float_format = "{:.2f}".format


# In[38]:


#내림차순 정렬
fi_df.sort_values(by="특성중요도", ascending=False, inplace=True)
# by= 기준이 되는 컬럼
fi_df[fi_df.values>=0.01]
#소수점2째 자리까지만 보기
#pd.options.display.float_format = "{:.2f}".format
#문자열로 타입 변환
fi_df.astype({"특성중요도":"str"})


# - oder_n 의 중요도가 압도적으로 높다
# - 중요다가 높은 특성을 먼저 고려해야 가장 효율적으로 분류할 수 있다.

# In[39]:


print(fi_df[fi_df.iloc[:]>=1].round(2).astype(str)+"%")


# In[ ]:




