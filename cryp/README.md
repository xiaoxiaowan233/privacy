1. Building HE Transformer

```
https://github.com/NervanaSystems/he-transformer
```

2. Simulated servers train a network

Only the split network is given in the document, and a network corresponding to the split network can be trained during binary training， Save in the upper folder"./inference"

If you are training with batchnormlization, some additional variables need to be saved because some of them are not supported by the encryption operation.

for example: Conv5_2 had added it to the floor，we should do like this

```python
           result,conv5_2_mean_run,conv5_2_var_run,conv5_1_mean_run,conv5_1_var_run,conv4_4_mean_run,conv4_4_var_run,conv4_3_mean_run,conv4_3_var_run,conv4_2_mean_run,conv4_2_var_run,acc = \
                   sess.run([y, conv5_2_mean, conv5_2_var, conv5_1_mean, conv5_1_var, conv4_4_mean, conv4_4_var, conv4_3_mean, conv4_3_var, conv4_2_mean, conv4_2_var,accuracy], feed_dict={x: test_image_batch, y_: test_label_batch})

```



3. example

clientRun.py ： This code simulates extracting features and can be run directly



pyclient.py： This code simulates encryption

serverRun.py: server

```
NGRAPH_ENCRYPT_DATA=1 NGRAPH_HE_SEAL_CONFIG=$HE_TRANSFORMER/configs/he_seal_ckks_config_N13_L8.json NGRAPH_TF_BACKEND=HE_SEAL python serverRun.py


NGRAPH_ENABLE_CLIENT=1 NGRAPH_ENCRYPT_DATA=1 NGRAPH_HE_SEAL_CONFIG=$HE_TRANSFORMER/configs/he_seal_ckks_config_N13_L8.json NGRAPH_TF_BACKEND=HE_SEAL python serverRun.py
```

