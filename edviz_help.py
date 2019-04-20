import pandas as pd

def pre_proc(df):
    df_cat = df.select_dtypes(exclude=['int64', 'float64'])
    df_dog = df.select_dtypes(include=['int64', 'float64'])
    
    if df_cat.shape[1]:
        df_cat = pd.get_dummies(df_cat, drop_first=False)
    
    df_dog = (df_dog - df_dog.min())/(df_dog.max()-df_dog.min())
    df_dog = df_dog - df_dog.mean()
    
    df = pd.concat([df_cat, df_dog], axis=1)
    
    df = df.fillna(df.mean())
    return df.dropna(axis=1, how='all')

def paired_pre_proc(df_train, df_test):
    df_cat_train = df_train.select_dtypes(exclude=['int64', 'float64'])
    df_dog_train = df_train.select_dtypes(include=['int64', 'float64'])
    
    if df_cat_train.shape[1]:
        df_cat_train = pd.get_dummies(df_cat_train, drop_first=False)
        
    df_cat_test = df_test.select_dtypes(exclude=['int64', 'float64'])
    df_dog_test = df_test.select_dtypes(include=['int64', 'float64'])
    
    if df_cat_test.shape[1]:
        df_cat_test = pd.get_dummies(df_cat_test, drop_first=False)
        
    df_dog_test = (df_dog_test - df_dog_train.min())/(df_dog_train.max()-df_dog_train.min())
    df_dog_test =  df_dog_test - df_dog_train.mean()
    
    df_dog_train = (df_dog_train - df_dog_train.min())/(df_dog_train.max()-df_dog_train.min())
    df_dog_train =  df_dog_train - df_dog_train.mean()
    
    df_train = pd.concat([df_cat_train, df_dog_train], axis=1)
    df_test = pd.concat([df_cat_test, df_dog_test], axis=1)
    
    df_train = df_train.fillna(df_train.mean())
    df_test = df_test.fillna(df_train.mean())
    
    df_train = df_train.dropna(axis=1, how='all')
    df_test = df_test[df_train.columns]
    
    return df_train, df_test

def pre_proc_no_norm(df):
    df_cat = df.select_dtypes(exclude=['int64', 'float64'])
    df_dog = df.select_dtypes(include=['int64', 'float64'])
    
    if df_cat.shape[1]:
        df_cat = pd.get_dummies(df_cat, drop_first=False)
    
    df = pd.concat([df_cat, df_dog], axis=1)
    
    df = df.fillna(df.mean())
    return df.dropna(axis=1, how='all')

def compare_clusters(df, cluster):
    df_stats = df.drop(     columns=['cluster', 'sub_cluster'], errors='ignore')
    cl_stats = cluster.drop(columns=['cluster', 'sub_cluster'], errors='ignore')
    
    df_stats = df_stats.applymap(lambda x: 1 if x == True else x)
    df_stats = df_stats.applymap(lambda x: 0 if x == False else x)
    cl_stats = cl_stats.applymap(lambda x: 1 if x == True else x)
    cl_stats = cl_stats.applymap(lambda x: 0 if x == False else x)
    
    df_stats = df_stats.describe().T
    cl_stats = cl_stats.describe().T
    
    print('There are %d samples (%f%%) in the first cluster.'
          % (cluster.shape[0], cluster.shape[0]/df.shape[0] * 100))
    print('%2.2f%% of the patients in the first cluster were admitted, compared to %2.2f%% in the second.'
          % (cl_stats.iloc[-1]['mean']*100, df_stats.iloc[-1]['mean']*100))
    
    diff = cl_stats - df_stats
    
    print('The first cluster is notable for these features above the mean:')
    display(diff['mean'].sort_values(ascending=False).head(10))
    print('The second cluster is notable for these features below the mean:')
    display(diff['mean'].sort_values(ascending=True).head(10))
    
def describe_cluster(df, cluster):
    df_stats = df.drop(     columns=['cluster', 'sub_cluster'], errors='ignore')
    cl_stats = cluster.drop(columns=['cluster', 'sub_cluster'], errors='ignore')
    
    df_stats = df_stats.applymap(lambda x: 1 if x == True else x)
    df_stats = df_stats.applymap(lambda x: 0 if x == False else x)
    cl_stats = cl_stats.applymap(lambda x: 1 if x == True else x)
    cl_stats = cl_stats.applymap(lambda x: 0 if x == False else x)
    
    df_stats = df_stats.describe().T
    cl_stats = cl_stats.describe().T
    
    print('There are %d samples (%f%%) in this peak.'
          % (cluster.shape[0], cluster.shape[0]/df.shape[0] * 100))
    print('%2.2f%% of the patients in this cluster were admitted, compared to %2.2f%% overall.'
          % (cl_stats.iloc[-1]['mean']*100, df_stats.iloc[-1]['mean']*100))
    
    diff = cl_stats - df_stats
    
    print('This cluster is notable for these features above the mean:')
    display(diff['mean'].sort_values(ascending=False).head(10))
    print('This cluster is notable for these features below the mean:')
    display(diff['mean'].sort_values(ascending=True).head(10))
    
def compare_cluster(df, cluster):
    df_stats = df.drop(     columns=['cluster', 'sub_cluster'], errors='ignore')
    cl_stats = cluster.drop(columns=['cluster', 'sub_cluster'], errors='ignore')
    
    df_stats = df_stats.applymap(lambda x: 1 if x == True else x)
    df_stats = df_stats.applymap(lambda x: 0 if x == False else x)
    cl_stats = cl_stats.applymap(lambda x: 1 if x == True else x)
    cl_stats = cl_stats.applymap(lambda x: 0 if x == False else x)
    
    df_stats = df_stats.describe().T
    cl_stats = cl_stats.describe().T
    
    print('There are %d samples (%f%%) in the first cluster.'
          % (cluster.shape[0], cluster.shape[0]/df.shape[0] * 100))
    print('%2.2f%% of the patients in the first cluster were admitted, compared to %2.2f%% in the second.'
          % (cl_stats.iloc[-1]['mean']*100, df_stats.iloc[-1]['mean']*100))
    
    diff = cl_stats - df_stats
    
    print('The first cluster is notable for these features above the mean:')
    display(diff['mean'].sort_values(ascending=False).head(10))
    print('The second cluster is notable for these features below the mean:')
    display(diff['mean'].sort_values(ascending=True).head(10))