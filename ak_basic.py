import autokeras as ak
import kerastuner as kt
print(f'Autokeras {ak.__version__}, Kerastuner {kt.__version__}')


early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', min_delta=0.01, patience=2, verbose=0
)

MAX_TRIALS = 4
EPOCHS = 2
TRAIN = False

# Initialize the text classifier.
clf = ak.TextClassifier(
    overwrite=True,  # overwrite any possible previous results
    max_trials=MAX_TRIALS,  #It only tries 2 models as a quick demo.
    seed=9)  # Set seed for randomizer to be able to reproduce results

if TRAIN:
    # Feed the text classifier with training data.
    clf.fit(
        x_train,
        y_train,
        epochs=EPOCHS,
        callbacks=[early_stop],
        verbose=2)