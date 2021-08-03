from pandas import read_csv
from matplotlib import pyplot
from numpy import std
from numpy import mean
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

data = 'Datasets-master/banknote_authentication.csv'
df = read_csv(data, header=None)
print(df.shape)
print(df.describe())
df.hist()
pyplot.show()

X, y = df.values[:, :-1], df.values[:, -1]
X = X.astype('float32')
print(X)
y = LabelEncoder().fit_transform(y)  # encode strings to integer
print(y)
# prepare k-fold cross validation
k_fold = StratifiedKFold(10)
scores = list()
for train_ix, test_ix in k_fold.split(X, y):
    X_train, X_test, y_train, y_test = X[train_ix], X[test_ix], y[train_ix], y[test_ix]  # split dataset into train and test distribution
    n_features = X.shape[1]
    print(n_features)

    # model architecture for MLP
    model = Sequential()
    model.add(Dense(10, activation='relu', kernel_initializer='he_normal', input_shape=(n_features,)))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()

    # model compilation
    model.compile(optimizer='adam', loss='binary_crossentropy')
    # Fitting the model
    history = model.fit(X_train, y_train, batch_size=32, epochs=50, verbose=0, validation_data=(X_test, y_test))
    print(history)
    y_hat = model.predict_classes(X_test)  # predict the test case
    print(y_hat)
    score = accuracy_score(y_test, y_hat)  # evaluating predictions
    print("Accuracy score %.3f" % score)
    scores.append(score)
    print('Mean Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
    pyplot.title('Learning Curves')
    pyplot.xlabel('Epoch')
    pyplot.ylabel('Cross Entropy')
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='val')
    pyplot.legend()
    # pyplot.show()
