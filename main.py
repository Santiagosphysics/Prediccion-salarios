from utils import pre

data = pre().creation_data('Salary_Data.csv')
target, features, names_f = pre().declaration_var(data, 'Salary')


one_gender, features = pre().one_hot('Gender', features)
one_education, features = pre().one_hot('Education Level', features)

features["Education Level_Bachelor's"], features = pre().replace("Education Level_Bachelor's", "Education Level_Bachelor's Degree", features )
features["Education Level_Master's"], features = pre().replace("Education Level_Master's", "Education Level_Master's Degree", features )
features["Education Level_PhD"], features = pre().replace("Education Level_PhD", "Education Level_phD", features )

features['Age'] = features['Age'].astype('int16')
features['Years of Experience'] = features['Years of Experience'].astype('int64')



one_job, features = pre().one_hot('Job Title', features)


print(target.info())


# print(features['Education Level_Masters Degree'])
# print(one_education.sum())

# print(one_gender)