#!/usr/bin/env python
# coding: utf-8

# ### 1.목표설정 문제정의
# - 타이타닉 승객 데이터를 학습해서 생존자와 사망자를 예측해보자!
# - 머신러닝 과정 이해

# ### 2.데이터 수집(데이터 로드)
# - 크롤링 or 자료수집
# - Kaggle 사이트에서 대회용 데이터 로드

# In[452]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[453]:


#csv를 불러오고 동시에 DataFrame 형태로 변환

train = pd.read_csv("data/train.csv", index_col = "PassengerId")
test = pd.read_csv("data/test.csv", index_col = "PassengerId")


# In[454]:


#데이터 프레임 전체 행 보기
pd.set_option("display.max_rows", None)

#데이터 프레임 전체 컬럼 보기
pd.set_option("display.max_columns", None)

#데이터 프레임 옵션값 초기화
pd.reset_option("display")


# ### 3.데이터 형태 체크 

# In[455]:


train.info()


# In[456]:


train.describe()


# In[457]:


test.info()


# In[458]:


test.describe()


# In[459]:


print(train.shape)
print(test.shape)


# - 결측치가 있는 컴럼이 확인됨
# - train : Age, Cabin Embarked
# - test : Age, Fare, Cabin

# In[460]:


train["Age"]


# In[461]:


train["Age"].describe()


# In[462]:


train["Cabin"].describe()


# In[463]:


#불리언 인덱싱을 활용하여 승객들을 찾아보자!

train[train["Cabin"]=="C23 C25 C27"]


# In[464]:


train["Embarked"].describe()


# In[465]:


train[train["Embarked"]=="S"]


# In[466]:


print(test["Age"].describe())
print(test["Fare"].describe())
print(test["Cabin"].describe())
print(test.shape)
print(test.info())


# ### 4.데이터 전처리
# - 데이터 결측치 확인
# - 결측치 채우기

# - Age 컬럼에 결측치를 평균으로 넣기에는 범위가 너무 넓다
# - 다른 컬럼들과의 상관관계를 이용하여 결측치를 채워보자!

# In[467]:


# corr : 상관계수를 파악하는 함수 -1 ~ 1 이다. correlation coefficient
# -1 (반비례) 1 (비례) 절대값이 클수록 상관관계가 크다
# 각 특성들간에 상관관계를 파악할 수 있다.

train.corr()


# - 행(row)를 보고 판단
# - Pclass 가 가장 상관계수가 높다.
# - 생존에 영향을 많이 준 컬럼인 성별을 활용해보자!

# In[468]:


# train 데이터프레임에 있는 값들로 피벗테이블을 만들어서 분석해보자
# 피벗테이블 : 컬럼들의 요약된 정보를 출력하는 테이블 형태

pt1 = train.pivot_table(values = "Age", # 데이터로 사용할 컬럼 지정
                       # 멀티인덱스 설정 (Pclass로 나눈 후 Sex로 나눔)
                       index = ["Pclass", "Sex"],
                        # 데이터 요약 시 사용하는 함수 지정
                        # mean: 평균 sum: 합계 count: 개수
                        aggfunc = "mean"
                       )
pt1


# In[631]:


# 멀티 인덱싱
pt1.loc[1, "female"]
pt1


# In[470]:


# Null값 체크
pd.isna(train["Age"]) # pandas 버전 (수치, 문자 데이터 판단 가능)


# In[471]:


np.isnan(train["Age"]) # numpy 버전 (수치형 데이터만 판단 가능)


# In[472]:


train["Age"].isnull() # 기본 제공 버전


# - 결측치를 채워주는 함수로 채워보자!

# In[473]:


def fill_age(data): # 매개변수는 train 혹은 test
    if pd.isna(data["Age"]):    
    # 위에서 만든 피벗테이블에서 멀티인덱싱한 나이 평균값을 리턴
    # 인덱스2개를 입력해서 보내면 해당 멀티인덱스 2개에
    # 해당하는 값을 시리즈 형태로 리턴한다.
        return pt1.loc[data["Pclass"], data["Sex"]]
    # 그렇지 않고 채워져 있다면 원래 값을 사용
    else:
        return data["Age"]


# In[474]:


# apply *문법 : 데이터 프레임에서 행렬에 복잡한 계산을 적용하고 싶을 때 사용
# apply 함수의 매개변수는 . 전에 들어간 train이 파라미터로 들어간다 
# train 데이터 프레임에서 fill_age 함수를 적용하는데 axis = 1 (행 방향, 위->아래) 로 적용

train["Age"] = train.apply(fill_age, axis = 1).astype("int64")
# 혹시 모를 에러를 방지하기 위해 astype 으로 형변환
train.info()


# In[475]:


# train 기준으로 fill_age 함수를 만들었기 때문에 사실상 사용하면 좋지 않다.
# 일반적으로 나누기 전에 미리 결측치를 해결하고 나누어야 한다.
# 이미 나누어져 있거나 어쩔 수 없는 경우는 데이터가 더 많은 train 을 기준으로 선언한 함수를 사용한다.

test["Age"] = test.apply(fill_age, axis = 1).astype("int64")
test.info()


# #### train Embarked 결측치 채우기
# - 결측치가 2개 뿐
# - 수치가 아닌 문자열

# In[476]:


train.info()


# In[477]:


# 결측치를 채우기 전에 빈도수 확인

train["Embarked"].value_counts()


# - 결측치가 2개 뿐이기에 가장 빈도수가 높은 "S" 를 사용

# In[478]:


# fiina() 비어있는 값에 파라미터를 채워주는 함수

train["Embarked"] = train["Embarked"].fillna("S")


# #### test Fare 결측치 채우기
# - 결측치 1개

# #  tip : 결측치가 너무 많다면 해당 컬럼을 안 쓰는 게 더 낫다 

# In[479]:


test["Fare"].describe()


# - 평균과 맥스값 차이가 크다
# - 결측치가 1개이므로 평균으로 채우는 편이 더 낫다
# - 상관관계 상관계수를 분석하여 채워보자!

# In[480]:


# correlation : 상관관계

test.corr()


# In[481]:


# train 데이터프레임에 있는 값들로 피벗테이블을 만들어서 분석해보자
# 피벗테이블 : 컬럼들의 요약된 정보를 출력할 수 있는 테이블 형태

pt2 = test.pivot_table(values = "Fare", # 데이터로 사용할 컬럼 지정
                       # 멀티인덱스 설정 (Pclass로 나눈 후 Sex로 나눔)
                       index = ["Pclass", "Sex"],
                        # 데이터 요약 시 사용하는 함수 지정
                        # mean: 평균 sum: 합계 count: 개수
                        aggfunc = "mean"
                       )
pt2


# In[482]:


# pivot 과 같은 기능
# pd.groupby (기준 두 개) ==> Pclass 와 Sex 는 index로 되고 Age만 컬럼이 된다.
train[["Pclass","Sex","Age"]].groupby(["Pclass","Sex"])


# In[483]:


# 멀티 인덱싱
pt2.loc[1, "male"]
pt2


# In[484]:


def fill_fare(data): # 매개변수는 train 혹은 test
    if pd.isna(data["Fare"]):
    # null 값 체크
    # 위에서 만든 피벗테이블에서 멀티인덱싱한 나이 평균값을 Siries 형태로 리턴
        return pt2.loc[data["Pclass"], data["Sex"]]
    # 그렇지 않고 채워져 있다면 원래 값을 사용
    else:
        return data["Fare"]


# In[485]:


test["Fare"] = test.apply(fill_fare, axis = 1).astype("float64")
test.info()


# In[486]:


# 불리언 인덱싱 
# 값이 잘 들어가서 전부 False

test[test["Fare"].isnull()]


# In[487]:


# 손으로 그냥 채워주기
# fillna() 비어있는 곳 채워넣기
# test[test["Fare"].isnull()].fillna(12.66)


# ### train, test Cabin 결측치 채우기
# - Cabin 유니크 값이 매우 많다
# - 문자+숫자
# - 과연 문자와 숫자중에 어떤 것이 더 중요한 데이터인가?

# In[488]:


train["Cabin"].value_counts()


# In[489]:


# 유니크 값이 너무 많다

train["Cabin"].unique()


# - 유니크 값이 너무 다양하고 많기에 고민
# - 숫자의 의미는 크지 않다고 판단하여 앞의 문자로 판단해보자!
# 
# ### 컬럼을 크게 변경하고 싶다면 컬럼을 하나 만들기

# In[490]:


# 결측치에 임의의 컬럼을 생성하여 "M" 이란 값을 넣어 분포도를 확인하자!

train["Deck"] = train["Cabin"].fillna("M")
test["Deck"] = test["Cabin"].fillna("M")
train["Deck"]


# In[491]:


# Deck 컬럼에서 맨앞에 영문자만 따오기
# 문자형에서 str[0] 0번째만 가지고 오세요~
train["Deck"] = train["Cabin"].fillna("M").str[0]
test["Deck"] = test["Cabin"].fillna("M").str[0]
train["Deck"]
print(train.shape)
print(test.shape)


# In[492]:


# 컬럼 삭제!

train.drop("Cabin", axis = 1, inplace = True)
test.drop("Cabin", axis = 1, inplace = True)


# In[493]:


train.info()


# ### 4. 탐색적 데이터 분석 (시각화) 
# - Deck컬럼 시각화
# - pivot 테이블과 비슷한 group by 사용해보기

# In[494]:


# groupby 그룹별로 데이터를 집계, 요약해주는 명령(그룹으로 묶어서 인덱스 설정)
# 사용할 컬럼 적고 그룹지을 컬럼만 따로 적기
train_deck = train[["Deck","Survived","Name"]].groupby(["Deck","Survived"]).count()
train_deck

# pivot 버젼!
# train.pivot_table(values="Name",
    # index = ["Deck", "Survived"],
    # aggfunc = "count"
#)


# In[495]:


# seaborn : 시각화 라이브러리
# matplotlib 에 비해 색상이 더 깔끔하고 성능이 더 좋음
# pandas 와의 호환성이 뛰어남
import seaborn as sns


# In[496]:


sns.countplot(data = train,
             x = "Deck",
             hue = "Survived"
             );


# - Cabin 컬럼에서 결측치였던 M에서 상대적으로 사람들이 많이 죽은 것이 확인
# - 다른 값들에 비해 월등히 높은 비율을 보이기 때문에 생존여부를 판단하는 데에 사용해도 좋을 것으로 판단 

# ### Pclass 컬럼 시각화
# - 비교적 상관계수가 높았기에 시각화를 통해 탐색

# In[497]:


sns.countplot(data = train,
             x = "Pclass",
             hue = "Survived"
             );


# ### Deck 과 Pclass 를 같이 시각화 

# In[498]:


sns.countplot(data = train,
             x = "Deck",
             hue = "Pclass"# x축에 pclass의 유니크 값들이 카운트
             );


# #### Sex, Embarked 컬럼 시각화
# - 각각 시각화를 진행하고 그 plot 들을 합쳐보자!

# In[499]:


# 남녀 성비를 볼 때 남자가 더 많이 탑승
# 남자가 더 많이 죽음
sns.countplot(data = train,
             x = "Sex",
             hue = "Survived"
             );


# In[500]:


# S 에서 가장 많은 사람들이 탑승
# S 에서 가장 많은 사람들이 죽음
sns.countplot(data = train,
             x = "Embarked",
            hue = "Survived"
             );


# In[501]:


# S에서 남자의 비율이 높다
sns.countplot(data = train,
             x = "Embarked",
            hue = "Sex"
             );


# #### Age 컬럼 시각화
# - Age 컬럼은 수치 범위가 너무 넓어서 countplot으로는 시각화가 힘듦

# In[502]:


sns.countplot(data = train,
             x = "Age",
            hue = "Survived"
             );


# In[503]:



plt.figure(figsize=(15,5))

sns.violinplot(data = train,
              x = "Sex",
              y = "Age",
              hue = "Survived",
              split = True # True 합 , False 분리
              );
plt.grid()

# 가운데 흰 점은 중앙값 Median
# 흰점 주위의 두꺼운 검은 선 사분위 범위
# 얇은 선 (95%) 신뢰구간

"수치값(실수)과 같이 연속된 데이터 이기에 값을 미리 예측 및 미분하여 기울기를 그리고 합쳐준다"


# - 20~30 대가 많이 탑승 및 사망
# - 여성과 어린 사람의 생존률이 높다

# #### Fare 컬럼 시각화 

# In[504]:


plt.figure(figsize=(15,5))

sns.violinplot(data = train,
              x = "Sex",
              y = "Fare",
              hue = "Survived",
              split = False # True 합 , False 분리
              );
plt.grid()


# In[505]:


# 시각화는 시각화만 해주기 때문에
# 데이터를 미리 잘라서 시각화
train["Fare"].describe()


# ##### * SibSp: 함께 탑승한 형제 또는 배우자 수
# ##### * Parch: 함께 탑승한 부모 또는 자녀 수
# - 공통점 : 가족
# - 하나의 컬럼으로 새롭게 생성하자!

# In[506]:


# Famile_Size 컬럼을 생성하고 Parch 와 SibSp 을 대입
train["Family_Size"] = train["Parch"]+train["SibSp"]+1
test["Family_Size"] = test["Parch"]+ test["SibSp"]+1


# In[507]:


sns.countplot(data = train,
             x = "Family_Size",
             hue = "Survived"
             );


# - 1명, 6인 이상일 때 사망률이 높고, 2~4인일 때는 생존률이 높다
# - 범주화를 해보자!
# - 1인 : Alone
# - 2~4인 : Small
# - 5~ : Large

# In[508]:


bins_size = [0,1,4,11]

# 0은 미포함 , 2부터 4까지, 5부터 11까지  : 처음은 미포함!
# 0~1 , 1~4 , 4~11 경계값

labels_size = ["Alone", "Small", "Large"]

# Framily_Group 컬럼을 새로 생성!
# cut() : bins, labels 속성
train["Family_Group"] = pd.cut(train["Family_Size"], bins = bins_size, labels = labels_size)
test["Family_Group"] = pd.cut(test["Family_Size"], bins = bins_size, labels = labels_size)


# In[509]:


train["Family_Size"]


# In[510]:


# 잘 적용이 되었는지 확인
train["Family_Group"]


# In[511]:


sns.countplot(data=train,
             x = "Family_Group",
             hue = "Survived"
             );


# ###   Text 데이터 다루기
# - Name 컬럼 분석
# - 중간 호칭만 분류하자

# In[512]:


train["Name"][1]


# In[513]:


# split() : 특정 문자를 기준으로 좌측과 우측, 인덱스 0과 1로 분리

train["Name"][1].split(",")


# In[514]:


train["Name"][1].split(",")[1].split(".")[0]


# In[515]:


# strip() : 좌우 공백을 지워주는 함수

train["Name"][1].split(",")[1].split(".")[0].strip()


# In[516]:


# 이름의 중간만 뽑아내는 함수

def split_name(data):
    
    return data.split(",")[1].split(".")[0].strip()


# In[517]:


# 행이나 열에는 바로 접근할 수 없기에 apply() 함수를 통해서 이를 적용할 수 있다.


# In[518]:


# train["Name"] 은 컬럼만 타겟팅했기에 시리즈 형태! axis = 1 처럼 축을 열로 따로 지정할 필요가 없음
# Name 컬럼과 Title 컬럼의 비교를 위해 새롭게 Title 컬럼 생성
train["Title"] = train["Name"].apply(split_name)
test["Title"] = test["Name"].apply(split_name)


# In[519]:


train["Name"]


# In[520]:


pd.set_option('display.max_rows', None)


# In[521]:


train["Title"]


# In[522]:


train["Title"].unique()


# In[523]:


plt.figure(figsize = (15,5))

sns.countplot(data = train,
             x = "Title",
             hue = "Survived"
             );


# In[524]:


plt.figure(figsize = (15,5))

# x , y 축 범위 설정 
# ylim : y축을 0~10까지 설정
plt.ylim(0,10) 

sns.countplot(data = train,
             x = "Title",
             hue = "Survived"
             );


# - 상대적으로 적은 개수의 컬럼들은 Other 로 묶어서 분석해보자!

# In[525]:


title = ['Mr', 'Mrs', 'Miss', 'Master', 'Rev','Don' , 'Dr', 'Mme',
       'Ms', 'Major', 'Lady', 'Sir', 'Mlle', 'Col', 'Capt',
       'the Countess', 'Jonkheer']


# In[526]:


len(title)


# In[527]:


convert_title = ['Mr', 'Mrs', 'Miss', 'Master', 'Rev']+["Other"]*12


# In[528]:


len(convert_title)


# In[529]:


# dict(zip()) 함수를 사용해서 리스트 형태인 title, convert_title 를 차례대로
# key, value 값으로 만드는 딕셔너리 생성


# In[530]:


title_dict = dict(zip(title, convert_title)) #key : value 값으로 매칭
title_dict


# In[531]:


# map() : 맵핑!

train["Title"] = train["Title"].map(title_dict)
train["Title"]


# - test 에는 train 에 없는 "Dona" 가 존재함
# - title_dict 를 그대로 사용하면 test 에는 적용이 되지 않음

# In[532]:


test["Title"].unique()


# In[533]:


title_dict["Dona"] = "Other"
title_dict


# In[534]:


test["Title"] = test["Title"].map(title_dict)
test["Title"]


# ### Ticket 컬럼 분석
# - 특징 파악이 어렵고 가공도 어려워 보임
# - 특별한 특징이나 인과과정이 안 보이고 가격 컬럼이 있기에 해당 컬럼은 삭제

# In[535]:


train["Ticket"].unique()


# In[536]:


print(train.info())
print(test.info())


# ### 가공이 끝났거나 필요하지 않은 컬럼은 삭제 (Ticket, Name, SipSb, Parch, Famile_Size)

# In[537]:


train.drop("Ticket", axis = 1, inplace = True)
test.drop("Ticket", axis = 1, inplace = True)

train.drop("Name", axis = 1, inplace = True)
test.drop("Name", axis = 1, inplace = True)

train.drop("Parch", axis = 1, inplace = True)
test.drop("Parch", axis = 1, inplace = True)

train.drop("Family_Size", axis = 1, inplace = True)
test.drop("Family_Size", axis = 1, inplace = True)


train.drop("SibSp", axis = 1, inplace = True)
test.drop("SibSp", axis = 1, inplace = True)


# ###  5. 모델링 (가공, 선택, 학습, 평가)
# - 문제 정답 데이터로 분리

# In[538]:


#y_train = train["Survived"] # 정답
#X_train = train.drop("Survived", axis = 1) # 문제 (정답 컬럼을 뺀 전체)

X_train = train.loc[:,'Pclass':] # 2차원적
y_train = train.loc[:,'Survived'] #1차원적 정답 데이터
X_test = test #테스터 데이터


# - 문자형 데이터를 숫자화

# In[539]:


# one hot 인코딩

X_train = pd.get_dummies(X_train) # 유니크값에 매칭되는 더미컬럼들을 생성하여 해당 값이 있는 컬럼만 1 표기 나머지는 0
X_test = pd.get_dummies(X_test)


# In[540]:


print(X_train.shape)
print(X_test.shape)


# In[541]:


# 어떤 컬럼이 다른지 눈으로 확인해보자
print(X_train.columns)
print(X_test.columns)


# In[542]:


# 어떤 컬럼이 다른지 찾아보자
# 같은 값이 있다면 빼져서 사라지고 차이만 남는다

set(X_train.columns) - set(X_test.columns)


# - 기계를 학습시키기 위해서 "Deck_T" 컬럼 생성 및 컬럼별 순서 정렬

# In[543]:


# 원핫인코딩을 사용 전 NaN 값이 들어가지 않도록 0으로 셋팅

X_test["Deck_T"] = 0
X_test


# In[544]:


tmp = X_train["Deck_T"] # Deck_T 추출을 위해 임시적 변수 tmp 에 잠시 넣어둔다


# In[545]:


X_train.drop("Deck_T", axis = 1, inplace = True) # inplace : 실시간 반영 여부 / axis = 1 세로 열을 축으로 접근 


# In[546]:


X_train = pd.concat([X_train, tmp], axis = 1) # 뒤로 옮겨서 붙이기


# In[547]:


set(X_train.columns) - set(X_test.columns)
print(X_train.shape)
print(X_test.shape)


# ### 모델 선택 

# In[548]:


# DecisionTree
from sklearn.tree import DecisionTreeClassifier
# KNN 
from sklearn.neighbors import KNeighborsClassifier
# 교차검증
from sklearn.model_selection import cross_val_score


# In[549]:


# 디시전트 모델
tree_model = DecisionTreeClassifier(max_depth = 3)


# In[550]:


# train 데이터를 디시전트 모델로 교차검증하여 정확도를 판단해보자!
# 교차검증을 활용하여 학습부터 평가까지 한 번에 진행

result_DT = cross_val_score(tree_model,
                           X_train, y_train,
                           cv = 5 # 몇개의 데이터 세트로 나눌지 설정
                           ) 
result_DT
print(X_train.shape)
print(X_test.shape)


# In[551]:


result_DT.mean()


# ### KNN모델 적용

# In[552]:


knn_model = KNeighborsClassifier(n_neighbors = 3)


# In[553]:


result_KNN = cross_val_score(knn_model,
                           X_train, y_train,
                           cv = 5 # 몇개의 데이터 세트로 나눌지 설정
                           ) 
result_KNN


# In[554]:


result_KNN.mean()


# - KNN 모델에 Scaler 적용해보기!

# In[555]:


from sklearn.preprocessing import StandardScaler


# In[556]:


scaler = StandardScaler()


# In[557]:


scaler.fit(X_train)


# In[558]:


transform_X_train = scaler.transform(X_train)
transform_X_train


# In[559]:


print(X_train.shape)
print(y_train.shape)
print(X_test.shape)

set(X_train.columns) - set(X_test.columns)
#transform_X_test = scaler.transform(X_test)
#transform_X_test


# In[560]:


# train 데이터를 KNN모델로 스케일링 후 교차검증을 적용해서 적황도를 판단해보자!

result_KNN_SS = cross_val_score(knn_model,
                                transform_X_train,
                                y_train,
                                cv = 5
                               )
result_KNN_SS


# - KNN 모델은 거리를 기반으로 하기 때문에 스케일링 후 약 6%p 차이가 발생할 수 있다

# In[561]:


result_KNN_SS.mean()


# ### Kaggle 제출용 파일을 만들어보자! 

# In[562]:


# gender_submission.csv 파일을 불러와서 해당 양식으로 정답을 예측해보자
result_submission = pd.read_csv("data/gender_submission.csv")
result_submission


# In[563]:


tree_model.fit(X_train, y_train) # 모델 학습
pre = tree_model.predict(X_test) # 학습된 모델로 예측해서 pre변수에 담기
pre


# In[564]:


# result_submission 의 Survived 컬럼의 모델로 예측한 결과인 pre 값을 갱신해줌

result_submission["Survived"] = pre


# In[565]:


result_submission


# In[566]:


# 최종 결과를 csv 파일로 저장 후 내보내기

result_submission.to_csv("submission_tree.csv", index = False)


# # Ensemble 모델 적용 

# In[567]:


# voting, random forest, adaboost, gradientboost
from sklearn.ensemble import VotingClassifier,RandomForestClassifier,AdaBoostClassifier,GradientBoostingRegressor


# - Random Forest

# In[568]:


# 모델 객체 생성
# bagging 방식
# estimators = 예측기 몇 개 만들래? default = 100
forest_model = RandomForestClassifier(n_estimators = 100,
                                     random_state = 11)
forest_model


# In[569]:


# 교차 검증의 성능점수 확인
# 교차검증 : 모델의 일반화 성능을 측정할 때 사용!
# 교차검증법은 정답까지 있어야 사용이 가능하다! **
from sklearn.model_selection import cross_val_score


# In[570]:


# cv = 몇 번 검증할래?
# 5번의 검증 정확도 중에서 평균이 어느정도인 모델인가?
# 개별적으로 나온 정확도들의 차이는 얼마일까?
cross_val_score(forest_model, X_train, y_train, cv = 5)

# 5번의 검증의 평균치를 구해보자!
cross_val_score(forest_model, X_train, y_train, cv = 5).mean()


# - Adaboosting 

# In[571]:


# 객체 생성
# estimators = 예측기 몇 개 만들래? default = 50
ada_model = AdaBoostClassifier(n_estimators = 100, random_state = 11)
ada_model


# In[572]:


# 교차 검증 
cross_val_score(ada_model, X_train, y_train, cv = 5).mean()


# - Voting

# In[573]:


# 객체 생성
voting_model = VotingClassifier(estimators = [("Knn1",knn_model),
                                            ("tree1", tree_model),
                                            ("forest1", forest_model),
                                            ("ada1", ada_model)], 
                     voting = "soft")


# In[574]:


# 교차 검증
cross_val_score(voting_model, X_train, y_train, cv = 5).mean()


# ### XGboost, LightGBM 설치 (외부 Lib)
# - 오차가 줄어드는 방향
# - 지니 불순도가 적은 방향
# - 최종적으로 평균을 구한다
# - 더 이상 배울 게 없다면 학습을 멈춘다

# In[575]:


get_ipython().system('pip install xgboost')


# In[576]:


get_ipython().system('pip install lightgbm')


# In[577]:


# 불러오기
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


# ### XG부스팅 

# In[578]:


xg_model = XGBClassifier(n_estimators = 100,
                       random_state = 11)


# In[579]:


# 최적의 파라미터 찾기 >> grid search
from sklearn.model_selection import GridSearchCV


# ### LightGBM 

# In[580]:


lgb_model = LGBMClassifier(n_estimators = 1000,
              random_state = 11,
              learning_rate = 0.1) # 가중치 설정 


# In[581]:


# 교차검증 정확도 평균 확인
xg1 = cross_val_score(xg_model, X_train, y_train, cv = 5)
lgb = cross_val_score(lgb_model, X_train, y_train, cv = 5)

print(xg1.mean())
print(lgb.mean())


# In[582]:


# 데이터에 따라 단순 모델이 성능이 좋을 수 있으나 대체로 Ensemble 성능이 좋다


# ### SVM 의 종류
# - Linear SVC : 선형 분류
# - Linear SVR : 선형 회귀
# - SVC : 비선형 분류
# - SVR : 비선형 회귀

# In[583]:


# 선형인지 비선형인지 그래프를 직접 그려봐야 알 수 있다.


# In[584]:


from sklearn.svm import LinearSVC, SVC


# In[585]:


# C 값 : 규제 >> 작을수록 규제가 강함

linearSVC = LinearSVC(C = 0.01, random_state = 0)
svc = SVC(C = 0.01, random_state = 0)


# In[586]:


linearSVC.fit(X_train, y_train)
linearSVC.score(X_train, y_train)


# In[587]:


#SVC 는 GAMMA 값이 필요하다
svc.fit(X_train, y_train)
svc.score(X_train, y_train)


# - 스케일링 후 SVM 적용해보자!

# In[588]:


# import sklearn.preprocessing import StandardScaler

sc = StandardScaler()

# fit 은 항상 train 데이터에만!
sc.fit(X_train)

# transform 은 학습이 끝난 train 과 test 에 적용!
X_train_scale = sc.transform(X_train)

linearSVC.fit(X_train_scale, y_train)

linearSVC.score(X_train_scale, y_train)


# ## GridSearch
# - 모델에서 파라미터의 최적값을 찾는 함수

# In[620]:


# GridSearchCV : 교차검증까지 해주는 함수
from sklearn.model_selection import GridSearchCV


# - best_params_ : 최고의 파라미터 값
# - best_score_ : 파라미터 값 점수
# - best_estimator_ : 전체 파라미터

# In[626]:


# 사용할 파라미터의 값 리스트
# key 값은 원래 파라미터의 이름과 동일해야 한다.
# values 에는 적용할 파라미터의 값들을 입력

param_grid = {"C":[0, 0.01, 0.0001, 1, 10, 100, 1000]}

linear2 = LinearSVC()

# cv : 교차 검증 시 fold 수 (묶음 개수)
gs = GridSearchCV(linear2, param_grid, cv = 5)

gs.fit(X_train_scale, y_train)

gs.score(X_train_scale, y_train)


# In[627]:


# 교차검증으로 인해 정확도는 떨어지지만 과대적합을 줄일 수 있어 일반화에 가까워짐


# In[628]:


gs.best_params_


# In[629]:


gs.best_score_


# In[630]:


gs.best_estimator_

