import os
import tensorflow as tf

from CCR import cnn_lstm_ctc_ocr
from CCR.eval_model import EvaluateModel
import CCR.utils as utils

from config import config

FLAGS = utils.FLAGS
configure = config["default"]


# Transform float tensor to image bitstring
def postprocess_float_tensor_to_bitstring(output_tensor):
    # Convert to uint8 tensor
    # output_tensor = (output_tensor + 1.0) / 2.0
    output_tensor = tf.image.convert_image_dtype(output_tensor, tf.uint8)

    # Remove the batch dimension
    output_tensor = tf.squeeze(output_tensor, [0])

    # Transform uint8 tensor to bitstring
    output_bytes = tf.image.encode_jpeg(output_tensor)
    output_bytes = tf.identity(output_bytes, name="output_bytes")
    return output_bytes


def image_decode(image_string):
    # Transform bitstring to uint8 tensor
    decoded_image = tf.image.decode_jpeg(image_string, dct_method='INTEGER_ACCURATE')

    reshape_input_tensor = tf.reshape(decoded_image, [configure.input_size_h, configure.input_size_w, 3])

    # rgb to bgr
    bgr_input_tensor = tf.reverse(reshape_input_tensor, axis=[-1])

    # Convert to float32 tensor
    input_tensor = tf.cast(bgr_input_tensor, dtype=tf.float32)

    # rescale to interval of [0, 1]
    img_input_placeholder = tf.math.divide(input_tensor, tf.constant(255.))

    return img_input_placeholder


if __name__ == "__main__":
    with tf.get_default_graph().as_default():
        # send image by base64
        image_string_list = tf.placeholder(tf.string, shape=[None, ], name='image_string')

        batch_input_tensor = tf.map_fn(image_decode, image_string_list, dtype=tf.float32)

        tfconfig = tf.ConfigProto()
        tfconfig.gpu_options.allow_growth = True  # maybe necessary, used to avoid cuda initialize error
        tfconfig.allow_soft_placement = True  # maybe necessary, used to avoid cuda initialize error
        # tfconfig.log_device_placement = True  # print message verbose
        with tf.Session(config=tfconfig) as sess:
            # # model of cnn lstm ctc
            cnn_lstm_ctc = EvaluateModel()
            ocr_model = cnn_lstm_ctc_ocr.LSTMOCR('eval', inputs=batch_input_tensor)
            ocr_model.build_graph()
            if tf.gfile.IsDirectory(configure.ccr_checkpoint_path):
                checkpoint_file = tf.train.latest_checkpoint(configure.ccr_checkpoint_path)
            else:
                checkpoint_file = configure.ccr_checkpoint_path
            ocr_cnn_scope_to_restore = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='cnn')
            ocr_lstm_scope_to_restore = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='lstm')
            ocr_stn_scope_to_restore = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='stn-1')
            ocr_restore = tf.train.Saver(
                ocr_cnn_scope_to_restore + ocr_lstm_scope_to_restore + ocr_stn_scope_to_restore)
            ocr_restore.restore(sess, checkpoint_file)

            ccr_dense_decoded = ocr_model.dense_decoded

            # tf server configure
            export_path_base = configure.ccr_export_dir
            export_path = os.path.join(
                tf.compat.as_bytes(export_path_base),
                tf.compat.as_bytes(str(len(os.listdir(configure.ccr_export_dir)))))
            print('Exporting trained model to', export_path)
            builder = tf.saved_model.builder.SavedModelBuilder(export_path)

            # define Input tensor info
            ccr_img_tensor_info_input_bytes = tf.saved_model.utils.build_tensor_info(image_string_list)

            # define Output tensor info
            ccr_dense_decoded_tensor_info_output = tf.saved_model.utils.build_tensor_info(ccr_dense_decoded)

            # create signature
            ccr_prediction_signature = (
                tf.saved_model.signature_def_utils.build_signature_def(
                    inputs={'image_bytes': ccr_img_tensor_info_input_bytes
                            },
                    outputs={
                        "ccr_dense_decoded": ccr_dense_decoded_tensor_info_output,

                    },
                    method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

            builder.add_meta_graph_and_variables(
                sess, [tf.saved_model.tag_constants.SERVING],
                signature_def_map={
                    'ccr': ccr_prediction_signature})

            # export model
            builder.save(as_text=True)
            print('Done exporting!')
