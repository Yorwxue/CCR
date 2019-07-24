import os
import cv2
import tensorflow as tf

from config import config
from app.api_v0_0_0.utilities import ccr_decode
from CCR.polyCrop import ratioImputation

configure = config["default"]


if __name__ == "__main__":
    model_name = "ccr"
    export_dir = os.path.join(configure.ccr_export_dir,
                              str(len(os.listdir(configure.ccr_export_dir))-1))  # model version

    image_path = os.path.abspath(os.path.join(__file__, "../..", "dataset", "123.jpg"))
    image = cv2.imread(image_path)

    image = ratioImputation(image, target_ration=(configure.input_size_h, configure.input_size_w))

    # bgr to rgb
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # image to bytes
    input_bytes = cv2.imencode('.jpg', image)[1].tostring()

    tfconfig = tf.ConfigProto()
    tfconfig.gpu_options.allow_growth = True  # maybe necessary
    tfconfig.allow_soft_placement = True  # maybe necessary
    with tf.Session(graph=tf.Graph(), config=tfconfig) as sess:
        tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], export_dir)

        print("exported %s model from %s" % (model_name, export_dir))

        # show nodes
        # graph = tf.get_default_graph()
        # for i in graph.get_operations():
        #     print(i.name)

        ccr_str = sess.run("dense_decoded:0", {"image_string:0": [input_bytes]})
        # ccr_str = sess.run("dense_decoded:0", {"image_placeholder:0": image})  # input as image for testing SaveModel

        # response_image_bytes = base64.b64decode(img_str)
        # with open('cc.png', 'wb') as f:
        #     f.write(response_image_bytes)

        texts = ccr_decode(ccr_str)
        print(texts[0])
