import tensorflow as tf
import numpy as np
import tensorflow.tools.graph_transforms as graph_transforms
import imagenet_lab as word_label
import argparse

INPUTS = 'input'
OUTPUTS = 'predict'
OPTIMIZATION = 'strip_unused_nodes remove_nodes(op=Identity, op=CheckNumerics) fold_constants(ignore_errors=true) fold_batch_norms fold_old_batch_norms'


# todo if the image shape is not 224,224,3, we need to add the preprocess
# currently, we just downsample and resize to desired shape
def preprocess(input):
    import cv2
    output = cv2.resize(input, (224, 224), interpolation=cv2.INTER_CUBIC)
    return output

def inference(image_path,input_graph):
    #read the image in jpeg format
    img = tf.read_file(image_path)
    input_op = tf.image.decode_jpeg(img, channels=3)
    sess = tf.Session()
    with sess.as_default():
        input = sess.run(input_op)
        if input.shape[0] != 224 or input.shape[1] != 224 or input.shape[2] != 3:
            #if the shape is not 224*224*3, preprocess the image, such as: resize
            input = preprocess(input)

        input = input.reshape(1, 224, 224, 3)

    #config the inference graph config
    infer_config = tf.ConfigProto()
    infer_config.intra_op_parallelism_threads = 26
    infer_config.inter_op_parallelism_threads = 1
    infer_config.use_per_session_threads = 1

    #read the pb model
    infer_graph = tf.Graph()
    with infer_graph.as_default():
      graph_def = tf.GraphDef()
      with tf.gfile.FastGFile(input_graph, 'rb') as input_file:
        input_graph_content = input_file.read()
        graph_def.ParseFromString(input_graph_content)

      output_graph = graph_transforms.TransformGraph(graph_def,
                                         [INPUTS], [OUTPUTS], [OPTIMIZATION])
      # for node in output_graph.node:
      #     print("name:{}   op:{}".format(node.name,node.op))

      tf.import_graph_def(output_graph, name='')

    # Definite input and output Tensors for detection_graph
    input_tensor = infer_graph.get_tensor_by_name('input:0')
    output_tensor = infer_graph.get_tensor_by_name('predict:0')
    infer_sess = tf.Session(graph=infer_graph, config=infer_config)

    predictions = infer_sess.run(output_tensor,
                                  {input_tensor: input})
    print(np.argmax(predictions))
    print("This image belong to : \"{}\"".format(word_label.label[np.argmax(predictions)-1]))

if __name__ == "__main__":
    import os, sys
    parser = argparse.ArgumentParser()
    parser.add_argument('-i',"--input_image", help="the path of the input image",dest='input',default=os.path.join(sys.path[0],"image/mouse.jpeg"))
    parser.add_argument('-m',"--input_model", help="the path of the input pb model",dest='model',default=os.path.join(sys.path[0],"pb/freezed_resnet50.pb"))
    args = parser.parse_args()
    inference(args.input,args.model)