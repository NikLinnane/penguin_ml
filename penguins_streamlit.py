import streamlit as st
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.title('Penguin Classifier')
st.write("This app allows you to predict the species of a penguin using our pretrained "
         "model or one trained on your own dataset")

# give the user the option to upload their own data to train a new model but if not use ours as default
penguin_file = st.file_uploader('Upload your own penguin data')

if penguin_file is None:
    # load pretrained model and mapping to target feature
    penguin_df = pd.read_csv('penguins.csv')
    rf_pickle = open('random_forest_penguin.pickle', 'rb')
    map_pickle = open('target_penguin.pickle', 'rb')
    rfc = pickle.load(rf_pickle)
    unique_penguin_mapping = pickle.load(map_pickle)
    rf_pickle.close()
    map_pickle.close()
else:
    penguin_df = pd.read_csv(penguin_file)
    penguin_df.dropna(inplace=True)
    target = penguin_df['species']
    features = penguin_df[['island', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex']]
    features = pd.get_dummies(features)
    target, unique_penguin_mapping = pd.factorize(target)
    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.8)
    rfc = RandomForestClassifier(random_state=15)
    rfc.fit(x_train, y_train)
    y_pred = rfc.predict(x_test)
    score = round(accuracy_score(y_pred, y_test), 2)
    st.write('We trained a Random Forest model on your data which has an accuracy score of {}. '
             'Use the imputs below to predict on your model!'.format(score))

# add user input features but wrap in a form function so it only runs when the button is pressed
with st.form('user_inputs'):
    island = st.selectbox('Penguin Island', options=['Biscoe', 'Dream', 'Torgerson'])
    sex = st.selectbox('Sex', options=['Female', 'Male'])
    bill_length = st.number_input('Bill Length (mm)', min_value=0)
    bill_depth = st.number_input('Bill Depth (mm)', min_value=0)
    flipper_length = st.number_input('Flipper Length (mm)', min_value=0)
    body_mass = st.number_input('Body Mass (g)', min_value=0)
    st.form_submit_button()

# initialize features and encode based on user input
island_biscoe, island_dream, island_torgerson = 0, 0, 0
if island == 'Biscoe':
    island_biscoe = 1
elif island == 'Dream':
    island_dream = 1
elif island == 'Torgerson':
    island_torgerson = 1

sex_female, sex_male = 0, 0
if sex == 'Female':
    sex_female = 1
elif sex == 'Male':
    sex_male = 1

# predict on the user's input data
new_prediction = rfc.predict([[bill_length, bill_depth, flipper_length, body_mass,
                               island_biscoe, island_dream, island_torgerson, sex_female, sex_male]])
prediction_species = unique_penguin_mapping[new_prediction][0]

# provide insight into the prediction
st.write('We predict that your penguin is of the {} species!'.format(prediction_species))
st.subheader("Predicting your penguin's species:")
st.write('We used a machine learning (Random Forest) model to predict the species and the features used in this '
         'prediction are ranked by relative importance below')
st.image('feature_importance.png')

st.write('Below are the histograms for each continuous variable separated by penguin species. Ther vertical '
         'line represents your inputted value')

fig, ax = plt.subplots()
ax = sns.displot(x=penguin_df['bill_length_mm'],
                 hue=penguin_df['species'])
plt.axvline(bill_length)
plt.title('Bill Length by Species')
st.pyplot(ax)

fig, ax = plt.subplots()
ax = sns.displot(x=penguin_df['bill_depth_mm'],
                 hue=penguin_df['species'])
plt.axvline(bill_length)
plt.title('Bill Depth by Species')
st.pyplot(ax)

fig, ax = plt.subplots()
ax = sns.displot(x=penguin_df['flipper_length_mm'],
                 hue=penguin_df['species'])
plt.axvline(bill_length)
plt.title('Flipper Length by Species')
st.pyplot(ax)

fig, ax = plt.subplots()
ax = sns.displot(x=penguin_df['body_mass_g'],
                 hue=penguin_df['species'])
plt.axvline(bill_length)
plt.title('Body Mass by Species')
st.pyplot(ax)