import tensorflow as tf
from google.protobuf import text_format
from tensorflow.python.platform import gfile

def pbtxt_to_graphdef(filename):
  with open(filename, 'r') as f:
    graph_def = tf.GraphDef()
    file_content = f.read()
    text_format.Merge(file_content, graph_def)
    tf.import_graph_def(graph_def, name='')
    tf.train.write_graph(graph_def, 'pbtxt/', 'protobuf.pb', as_text=False)

def graphdef_to_pbtxt(filename): 
  filenametoParse = filename+'.pb'
  with gfile.FastGFile(filenametoParse,'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')
    tf.train.write_graph(graph_def, 'pbtxt/', filename+'.pbtxt', as_text=True)
  return


graphdef_to_pbtxt('FaceSet')
graphdef_to_pbtxt('FaceSet2') 
graphdef_to_pbtxt('face_corrector') 

 # here you can write the name of the file to be converted
# and then a new file will be made in pbtxt directory.