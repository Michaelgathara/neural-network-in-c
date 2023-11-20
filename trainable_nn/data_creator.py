import pandas as pd
import numpy as np
import datetime

# https://stackoverflow.com/a/415519/11009561
now = datetime.datetime.now()


num_samples = 1000000
num_input_nodes = 5


np.random.seed(0) 
inputs = np.random.rand(num_samples, num_input_nodes)


weights = np.array([0.2, 0.3, 0.5, 0.1, 0.4])  
targets = np.dot(inputs, weights) + np.random.normal(0, 0.1, num_samples)

data = pd.DataFrame(inputs, columns=[f'Input_{i+1}' for i in range(num_input_nodes)])
data['Target'] = targets


csv_file_path = f'{hash(now.time())}.csv'
data.to_csv(csv_file_path, index=False)