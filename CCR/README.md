# CCR netowrk : *C*onvolutional *C*TC *R*ecurrent networks
+ This is an implement of Optical Character Recognition(ocr).
## For other task
+ We using this project to recognite the number of car license plate, if you have other targets, you can edit "charset" in line 50 of "CCR/CCR/utils.py", such as:
```python
charset = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
```
## Train
+ Specific directory of your data:
```python
tf.app.flags.DEFINE_string('data_dir', './imgs/', 'the data dir')
```

## Evaluate
+ Testing code can be found in "CCR/CCR/eval_model.py".
+ You can reference the code in the bottom of "CCR/CCR/eval_model.py".
---

## Release History
+ Branch "master":
    + Implement a Optical Character Recognition(ocr), based on cnn, lstm with ctc(Connectionist Temporal Classification).
+ Branch "fc":
    + Without time series model, just using full connective layer to generate prediction.
+ Branch "qrnn":
    + Using qrnn to replace lstm for time series.
+ Branch "master":
    + Built a celery+flask based service server.
## Authors

## Reference

