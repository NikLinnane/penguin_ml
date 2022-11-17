import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

# load data
penguin_df = pd.read_csv('penguins.csv')

# simple cleaning
penguin_df.dropna(inplace=True)

# define target & features
target = penguin_df['species']
features = penguin_df[['island', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex']]

# dummy
features = pd.get_dummies(features)

# train, test, split
target, uniques = pd.factorize(target)
x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.8)

# train model
rfc = RandomForestClassifier(random_state=15)
rfc.fit(x_train, y_train)
# predict
y_pred = rfc.predict(x_test)
# accuracy
score = accuracy_score(y_pred, y_test)
print('The accuracy score for our model us {}%'.format(round(score, 3)*100))

# pickle model to use in our app
rf_pickle = open('random_forest_penguin.pickle', 'wb')
pickle.dump(rfc, rf_pickle)
rf_pickle.close()
target_pickle = open('target_penguin.pickle', 'wb')
pickle.dump(uniques, target_pickle)
target_pickle.close()

fig, ax = plt.subplots()
ax = sns.barplot(rfc.feature_importances_, features.columns)
plt.title('Which features are the most important for predicting species?')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
fig.savefig('feature_importance.png')