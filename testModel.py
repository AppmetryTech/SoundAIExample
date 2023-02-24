import tensorflow as tf
import librosa
import os


def testData():
    """ The test path will contain the folder containing all the test files. Place only the test files in that fodler and nothing else.
    The catA, catB are the variables that will denote the 2 classes in your model. Ex: CatA would be cats and catB would be dogs
    Sampling rate of the audio files. Due to computations, all models are trained with 16000 Hz"""

    catA, catB = ["CatA", "CatB"]   #Replace CatA and catB in the list with your categories
    testPath = ""                   #Enter your test path
    sampling_rate = 16000           #Do not change
    outputDict = {}
    model = tf.lite.Interpreter(model_path="model.tflite")     #Enter your tflite path
    model.allocate_tensors()

    input_details = model.get_input_details()
    output_details = model.get_output_details()

    for file in os.listdir(testPath):
        data, sampling_rate = librosa.load(testPath + "/" + file, sr=sampling_rate)
        MFCCs = librosa.feature.mfcc(data, n_mfcc=13, sr=sampling_rate)
        ip = [MFCCs.T.flatten()]
        model.set_tensor(input_details[0]['index'], ip)
        model.invoke()
        y = model.get_tensor(output_details[0]['index'])[0][0]
        outputDict[file] = catA if y < .5 else catB
    return outputDict
