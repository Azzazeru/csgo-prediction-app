import sys
import numpy as np
import pickle

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')

with open('modelo_csgo.pkl', 'rb') as file:
    model = pickle.load(file)

input_values = sys.argv[1:]
input_values = list(map(float, input_values))

prediction = model.predict(np.array(input_values).reshape(1, -1))

print(prediction[0])