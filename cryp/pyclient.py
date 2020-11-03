
import pyhe_client
import time
import argparse

import numpy as np

import time

HEDataFile = "data.txt"
HELabelFile = "groundTruth.txt"

batch_size = 1024
newdata = np.loadtxt(HEDataFile, dtype=np.float32)

ground_truth = np.loadtxt(HELabelFile, dtype = np.float32).reshape(batch_size,100)


hostname = 'localhost'
port = 34000
client = pyhe_client.HESealClient(hostname, port, batch_size, newdata,False)

print('Sleeping until client is done')
while not client.is_done():
    time.sleep(1)

result = client.get_results()
result = np.array(result)
result = result.reshape(batch_size, 100)
print("result shape: ", result.shape)
y_label_batch = np.argmax(ground_truth, 1)
y_pred = np.argmax(result, 1)
correct_prediction = np.equal(y_pred, y_label_batch)
error_count = np.size(correct_prediction) - np.sum(correct_prediction)
test_accuracy = np.mean(correct_prediction)

print('Error count', error_count, 'of', batch_size, 'elements.')
print('Accuracy: %g ' % test_accuracy)
# print('Error count', error_count, 'of', batch_size, 'elements.')
# print('Accuracy: %g ' % test_accuracy)

# def main(FLAGS):
#     data = (1, 2, 3, 4)

#     port = 34000
#     batch_size = 1

#     client = pyhe_client.HESealClient(FLAGS.hostname, port, batch_size, data,
#                                       False)

#     while not client.is_done():
#         time.sleep(1)

#     results = client.get_results()

#     print('results', results)


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         '--hostname', type=str, default='localhost', help='Hostname of server')

#     FLAGS, unparsed = parser.parse_known_args()

#     print(FLAGS)
#     main(FLAGS)