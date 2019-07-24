# How to Create Tensorflow-serving
###### tags: `work`
[toc]
## Define Inputs of SaveModel
+ declare new placeholders as input of tf-serving
+ tf-serving will automatically doing ==web-saved== image-encoded/decoded, but you have to do it when using Savemodel without tf-serving.
+ if there are any operation for data preprocessing, using "tf.map_fn" function to deal with it.
    + method to using "tf.map_fn" function just like "map" function in python.
        ```python
        tf.map_fn(<function>, (<input1>, <input2>, ...), (<dtype of output1>, <dtype of output2>, ...))
        ```
        + Note that if the architectures of inputs and outputs aren't the same, the attribution of ==dtype== must be given.
        + The return tensors of input datas in the batch of "tf.map_fn" must have ths same shapes.
+ combine the input placeholder into model architecture
+ load model
+ Note that computing will cost more memory than load model, so the following setting of tensorflow session may be necessary
```python
tfconfig = tf.ConfigProto()
tfconfig.gpu_options.allow_growth = True
tfconfig.allow_soft_placement = True #  allow using some cpu
tfconfig.per_process_gpu_memory_fraction = 0.4
```
## Create Signature
### Parameter of SaveModel
```python
export_path_base = <path to save model>
export_path = os.path.join(
    tf.compat.as_bytes(export_path_base),
    tf.compat.as_bytes("VERSION_OF_MODEL")
)
builder = tf.saved_model.builder.SavedModelBuilder(export_path)
```
+ VERSION_OF_MODEL must be a string of integer(?
### Define Input/Output Tensor Info
+ such as: 
    ```python
    TENSOR_INFO = tf.saved_model.utils.build_tensor_info(PLACEHOLDER)
    ```
### Architecture of Signature
+ [official document](https://www.tensorflow.org/tfx/serving/signature_defs)
```python
SIGNATURE_DEF = (
    tf.saved_model.signature_def_utils.build_signature_def(
        inputs = {
            "KEY_OF_INPUT_TENSOR": INPUT_TENSOR_INFO_1,
                                    :
        },
        outputs = {
            "KEY_OF_RETURN_JSON": OUTPUT_TENSOR_INFO_1,
                                    :
        },
        method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
    )
)
```
+ Note: 
+ If input is byte string like encoded image, KEY_OF_INPUT_TENSOR most include =="_bytes"== at the end, such as "image_bytes"
### Build Signature:
```python
builder.add_meta_graph_and_variables(
    sess, [tf.saved_model.tag_constants.SERVING],
    signature_def_map={
        "SERVICE_NAME": SIGNATURE_DEF_1,
                        :
    }
)
```

+ "signature_def_map" allow applying multi-service once time
+ ==`SERVICE_NAME`== is the key of specific service which can be called by client.

### SaveModel
#### Save
```python
builder.save(ss_text=True)
```
#### Load
```python
tf.saved_model.loader.load(SESSION, [tf.saved_model.tag_constants.SERVING], EXPORT_DIRECTORY)
SESSION.run("OUTPUT_NODE_NAME", {"INPUT_NODE_NAME": INPUT_DATA})
```
+ OUTPUT_NODE_NAME: name of node in graph, such as "output_1:0".
+ INPUT_NODE_NAME: name of node in graph, such as "input_1:0".
+ Note that the substring ==":0"== in the node name is necessary.
+ If input data is image encoded by base64 , it must be [web-saved](#Define-Inputs-of-SaveModel).
+ You can get the input/output nodes by type in:
```python
metagraph = tf.saved_model.loader.load(sess, [tag_constants.SERVING], /YOUR/PATH/TO/SAVEDMODEL/)
inputs_mapping = dict(metagraph.signature_def["YOUR_SIGNATURE_NAME"].inputs)
outputs_mapping = dict(metagraph.signature_def["YOUR_SIGNATURE_NAME"].outputs)
```
## configure
+ model configure
```json
model_config_list: {
 config:{
     name: <"service name 1">,
     base_path: <"path to model directory">,
     model_platform: "tensorflow"
 },
 config:{
     name: <"service name 2">,
     base_path: <"path to model directory">,
     model_platform: "tensorflow"
 },
                 :
}
```
+ batch configure
```json
max_batch_size { value: 128 }
batch_timeout_micros { value: 1000 }
max_enqueued_batches { value: 64}
num_batch_threads { value: 6 }
```
+ start serving
    + type the following command in terminal to start tensorflow serving
    ```bash
    $ tensorflow_model_server --rest_api_port PORT_NUMBER --model_config_file=MODEL_CONFIG_PATH --enable_batching=true --batching_parameters_file=BATCH_CONFIG_PATH -- per_gpu_memory_fraction=RATIO_OF_USABLE_GPU
    ```
## How to Post Data and Get Response
### Query Format
+ if the data you want to send is an image, you must encoded it by base64 encoder, and the first key must include =="_bytes"==(as aforemention in ["Create Signature"](#Create-Signature)). You also need to give the second key =="b64"== to tell tensorflow serving this data need to be decoded, like the follow example.
+ row format (We prefer to use this format.)
```python
headers={"content-type": "application/json"}
body = {
    "signature_name": SERVICE_NAME,
    "instances": [
        # ex:
        # image 1
        {"image_bytes": {"b64": image_content1},
         "image_shape": image_shape1},
        # image 2
        {"image_bytes": {"b64": image_content2},
         "image_shape": image_shape2},
                        :
        # text data 1
        {"text": text_content1},
        # text data 2
        {"text": text_content2},
                   :
    ]
}
resp = requests.post(URL, data=json.dumps(body), headers=headers)
```
+ column format
```python
body = {
 "inputs": {
   "tag": ["foo", "bar"],
   "signal": [[1, 2, 3, 4, 5], [3, 4, 1, 2, 5]],
   "sensor": [[[1, 2], [3, 4]], [[4, 5], [6, 8]]]
 }
}
```
+ more detail of row/column format can be found [here](https://www.tensorflow.org/tfx/serving/api_rest)
+ you can also use "curl" command to test your tf-serving as following:
```bash
curl -X POST --data INPUT_DATA URL_OF_SERVICE

```
+ URL is the ip and port of the service, for example:
    + http://127.0.0.1:5000
+ if you using docker-compose, it should look like this:
    + http://SERVICE_NAME:5000
    + ==SERVICE_NAME== is defined in [model configure](#configure)
### response of tensorflow serving
+ if it works, tf-serving will return a json format string as following structure:
```python
resp["predictions"][INDEX_OF_DATA_INT_BATCH]["KEY_OF_RETURN_JSON"]
```
+ ==KEY_OF_RETURN_JSON== must match the defined key in [Architecture of Signature](#Architecture-of-Signature)
+ If you use column format, the json returned by tf-serving will look like this:
```python
resp["outputs"]["KEY_OF_RETURN_JSON"]
```
