# %% [markdown]
# # Explore here

# %%
# Your code here
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV
from sklearn.metrics import r2_score, median_absolute_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import xgboost as xgb
from statsmodels.stats.outliers_influence import variance_inflation_factor

# %% [markdown]
# GOAL : Predict the number of Population with Heart Disease
# 

# %%
df = pd.read_csv('https://raw.githubusercontent.com/4GeeksAcademy/regularized-linear-regression-project-tutorial/main/demographic_health_data.csv')
df

# %%
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 110)
df.head()

# %%
print(df.info())
print(f'''
      shape: {df.shape}''')

# %%
aux = list(df.columns[df.dtypes=='object'])
print(f'Columns dtypes == "object": {aux}')

# %%
unique_var = df.nunique()==1

if any(unique_var): 
    for col in unique_var.index[unique_var] : print(f"The variable {i} content only one category.")
else: print(f"There's no constant variable, or variable with only one category to clean.")

# %%
aux = df.nunique().sort_values().head()
aux1 = df.nunique().sort_values().tail()
print(f''' 
      Unique values sorted head(5):
      
      {aux}
      
      Unique values sorted tail(5)
      
      {aux1}
      ''')

# %%
aux = pd.DataFrame(list(df.columns))
print(aux)

# %%
aux = ['TOT_POP', '0-9', '19-Oct', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80+'] # descarto estas columnas para trabajar mejor con su equivalente en % (ya normalizado)
depured_df = df.drop(columns=aux)
depured_df = depured_df.drop(columns=['COUNTY_NAME', 'STATE_NAME']) # descarto estas tambien, si las necesito las tengo en df (pero ya tengo la columna "FIPS" y 'COUNTY FIPS')
aux1 = ['0-9 y/o % of total pop', '10-19 y/o % of total pop', '20-29 y/o % of total pop', '30-39 y/o % of total pop', '40-49 y/o % of total pop', 
        '50-59 y/o % of total pop', '60-69 y/o % of total pop', '70-79 y/o % of total pop', '80+ y/o % of total pop'] # a estas les voy a cambiar el nombre y las paso a valor absoluto
depured_df[aux1] = depured_df[aux1]/100
aux2 = '% ' #le agrego esto a los nombres de las col borradas para que se visibilize mejor
dic_aux = dict(zip(aux1, (aux2+i for i in aux[1:])))
depured_df.head()

# %%
aux = depured_df.isna().sum().sum()
if aux > 0 : print(f"There are {aux} nan values in the df") 
else : print(f"There are no nan values in the df")

# %%
# Elimino las columnas que entiendo que no me interesan para el objetivo: 
aux = ['White-alone pop', 'Black-alone pop', 'Native American/American Indian-alone pop', 'Asian-alone pop', 'Hawaiian/Pacific Islander-alone pop', 'Two or more races pop', 
'Less than a high school diploma 2014-18', 'High school diploma only 2014-18', "Some college or associate's degree 2014-18", "Bachelor's degree or higher 2014-18",
'POVALL_2018', 'CI90LBINC_2018', 'CI90UBINC_2018', 'Civilian_labor_force_2018', 'Employed_2018', 'Unemployed_2018', 'Median_Household_Income_2018', 'Total Population',
'Population Aged 60+', 'Percent of Population Aged 60+', 'county_pop2018_18 and older', 'anycondition_Lower 95% CI', 'anycondition_Upper 95% CI', 'anycondition_number',
'Heart disease_Lower 95% CI', 'Heart disease_Upper 95% CI','COPD_Lower 95% CI', 'COPD_Upper 95% CI', 'COPD_number', 
'diabetes_Lower 95% CI', 'diabetes_Upper 95% CI', 'diabetes_number', 'CKD_Lower 95% CI', 'CKD_Upper 95% CI', 'CKD_number', 'Urban_rural_code', 'N_POP_CHG_2018', 'GQ_ESTIMATES_2018',
'Obesity_Lower 95% CI', 'Obesity_Upper 95% CI', 'Obesity_number']
depured_df = depured_df.drop(columns=aux)


# %%
# Llevo los valores expresados en porcentajes a val abs
aux = ['% White-alone','% Black-alone','% NA/AI-alone','% Asian-alone','% Hawaiian/PI-alone','% Two or more races','R_birth_2018', 'R_death_2018', 'R_NATURAL_INC_2018',
       'R_INTERNATIONAL_MIG_2018', 'R_DOMESTIC_MIG_2018', 'R_NET_MIG_2018', 'Percent of adults with less than a high school diploma 2014-18', 'Percent of adults with a high school diploma only 2014-18',
       "Percent of adults completing some college or associate's degree 2014-18", "Percent of adults with a bachelor's degree or higher 2014-18", 
       'Family Medicine/General Practice Primary Care (2019)', 'PCTPOVALL_2018','PCTPOV017_2018','PCTPOV517_2018', 'Unemployment_rate_2018', 
       'Med_HH_Income_Percent_of_State_Total_2018', 'anycondition_prevalence', 'Obesity_prevalence', 'Heart disease_prevalence', 'COPD_prevalence', 
       'diabetes_prevalence', 'CKD_prevalence']
depured_df[aux] = depured_df[aux]/100

depured_df.head()

# %%
# Veo las columnas a normalizar 
aux3 = (depured_df > 1).any()
dep_df_noperc_columns = aux3[aux3==True].index.to_list()
dep_df_noperc_columns

# %%
columns_not_to_take = ['fips', 'STATE_FIPS', 'CNTY_FIPS']
dep_df_noperc_columns = [col for col in dep_df_noperc_columns if col not in columns_not_to_take]
dep_df_noperc_columns

# %%
# Dejo unicamente las columnas a trabajar, y escalo las que no son abs values de porcentajes
depured_df.drop(columns=columns_not_to_take, inplace=True)
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(depured_df[dep_df_noperc_columns])
depured_df[dep_df_noperc_columns] = df_scaled
depured_df


# %%
# Chequeo si quedaron columnas a normalizar
aux3 = (depured_df > 1).any()
dep_df_noperc_columns = aux3[aux3==True].index.to_list()
print(f'Columns to be normalized: {dep_df_noperc_columns}.')

# %%
# Variables independientes/dependiente
X = depured_df.drop(['Heart disease_number'], axis=1)
y = depured_df['Heart disease_number']

# Ajusta el modelo
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()

# Instancia de la influencia y obtención de la distancia de cook para cada observación
cooks_distance = model.get_influence().cooks_distance

# %%
plt.scatter(depured_df['Heart disease_number'], cooks_distance[0])
plt.xlabel('x')
plt.ylabel('Cooks Distance')
plt.show()

# %%
# Si la distancia es mayor de 4/n se considera observación influyente
pto_corte = 4/len(X)
pto_corte

# %%
# observaciones influyentes
len(np.where(cooks_distance[0]>pto_corte)[0])/len(X)*100

# %%
# descarto las influyentes
print(depured_df.shape)
depured_df.drop(np.where(cooks_distance[0]>pto_corte)[0], inplace=True)
depured_df.shape

# %%
# 80% del dataset para entrenamiento y un 20% para conjunto de test.
X = depured_df.drop(columns='Heart disease_number', axis=1)
y = depured_df['Heart disease_number']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %%
# xgb para conocer la importancia de las características
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42).fit(X_train, y_train)
df_imp = pd.DataFrame({'Feature':X_train.columns, 'Importance':xgb_model.feature_importances_*100})
df_imp = df_imp.sort_values(by='Importance', ascending=False)
df_imp

# %%
# Punto de corte, tomamos las superiores a 0.5%
df_imp[df_imp.Importance>=0.5].Feature.values

# %%
# Dejo las columnas del punto de corte
X_train = X_train[df_imp[df_imp.Importance>=0.5].Feature.values]
X_test = X_test[df_imp[df_imp.Importance>=0.5].Feature.values]

# %%
vif = pd.Series([variance_inflation_factor(X_train.corr().values, i) for i in range(X_train.corr().shape[1])], index=X_train.columns)
vif

# %%
X_corr = X_train.copy()
X_corr['y'] = y_train
corr = X_corr.corr()
corr.style.background_gradient(cmap='coolwarm').format(precision=3)

# %%
X_train.drop(['Total nurse practitioners (2019)', 'Total Specialist Physicians (2019)', 'ICU Beds_x'], axis=1, inplace=True)
X_test.drop(['Total nurse practitioners (2019)', 'Total Specialist Physicians (2019)', 'ICU Beds_x'], axis=1, inplace=True)

# %%
X_corr = X_train.copy()
X_corr['y'] = y_train
corr = X_corr.corr()
corr.style.background_gradient(cmap='coolwarm').format(precision=3)

# %%
m_lineal = LinearRegression().fit(X_train, y_train)
y_pred_linear_train = m_lineal.predict(X_train)
y_pred_linear_test = m_lineal.predict(X_test)

# %%
def get_metrics(yhat, y_test, yhat_train, y_train):
  metrics_train = r2_score(y_train, yhat_train), median_absolute_error(y_train, yhat_train), mean_absolute_percentage_error(y_train, yhat_train)*100
  metrics_test = r2_score(y_test, yhat), median_absolute_error(y_test, yhat), mean_absolute_percentage_error(y_test, yhat)*100
  metrics_diff = list(map(lambda x: x[1]-x[0], zip(metrics_train, metrics_test)))
  return pd.DataFrame(data=[metrics_train, metrics_test, metrics_diff], columns=['R2', 'Median AE', 'MAPE'], index=['Train set', 'Test set', 'Diferencia'])

# %%
get_metrics(y_pred_linear_test, y_test, y_pred_linear_train, y_train)

# %%
m_lineal.coef_

# %%
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.boxplot(m_lineal.coef_)
plt.title('Boxplot de los coeficientes de regresión lineal')
plt.subplot(1, 2, 2)
plt.hist(m_lineal.coef_,20)
plt.xlabel('Coeficientes')
plt.ylabel('Total de coeficientes para cada rango')
plt.title('Histograma de coeficientes')
plt.show()

# %%
# empezamos a trabajar con el modelo de Lasso
m_lasso = LassoCV(alphas=np.logspace(-6,6,10), cv=5, random_state=42, n_jobs=-1).fit(X_train, y_train)

# %%
y_pred_lasso_train = m_lasso.predict(X_train)
y_pred_lasso_test = m_lasso.predict(X_test)

# %%
get_metrics(y_pred_linear_test, y_test, y_pred_linear_train, y_train)

# %%
get_metrics(y_pred_lasso_test, y_test, y_pred_lasso_train, y_train)

# %%
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.boxplot(m_lasso.coef_)
plt.title('Boxplot de los coeficientes de regresión lineal')
plt.subplot(1, 2, 2)
plt.hist(m_lasso.coef_,10)
plt.xlabel('Coeficientes')
plt.ylabel('Total de coeficientes para cada rango')
plt.title('Histograma de coeficientes')
plt.show()

# %%
print(f'El porcentaje de variables canceladas es: {round(len(m_lasso.coef_[np.abs(m_lasso.coef_)==0])/len(m_lasso.coef_)*100,2)}%')
print(f'El modelo utiliza {len(m_lasso.coef_[np.abs(m_lasso.coef_)>0])} variables.')

# %%
# vamos con el ridge
m_ridge = RidgeCV(alphas=np.logspace(-6,6,10), cv=5).fit(X_train, y_train)

# %%
y_pred_ridge_train = m_ridge.predict(X_train)
y_pred_ridge_test = m_ridge.predict(X_test)

# %%
get_metrics(y_pred_ridge_test, y_test, y_pred_ridge_train, y_train)

# %%
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.boxplot(m_ridge.coef_)
plt.title('Boxplot de los coeficientes de regresión lineal')
plt.subplot(1, 2, 2)
plt.hist(m_ridge.coef_,10)
plt.xlabel('Coeficientes')
plt.ylabel('Total de coeficientes para cada rango')
plt.title('Histograma de coeficientes')
plt.show()

# %%
print(f'El porcentaje de variables canceladas es: {round(len(m_ridge.coef_[np.abs(m_ridge.coef_)==0])/len(m_ridge.coef_)*100,2)}%')
print(f'El modelo utiliza {len(m_ridge.coef_[np.abs(m_ridge.coef_)>0])} variables.')



