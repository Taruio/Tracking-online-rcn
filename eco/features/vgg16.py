import tensorflow.contrib.slim as slim
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow



class VGG16Net:
    def __init__(self):
        self._image = tf.placeholder(tf.float32, shape=[1, None, None, 3])
        tfconfig = tf.ConfigProto(allow_soft_placement=True)
        tfconfig.gpu_options.allow_growth = True
        self.sess = tf.Session(config=tfconfig)
        with self.sess.graph.as_default():
            self.avg_net, self.net = self.build_head()
            print('Loading VGG16 weights:...')
            variables = tf.global_variables()
            self.sess.run(tf.variables_initializer(variables, name='init'))
            var_keep_dic = self.get_variables_in_checkpoint("/home/lhc/Desktop/pyECO-master/data/vgg16.ckpt")
            # print(variables)
            variables_to_restore = self.get_variables_to_restore(variables, var_keep_dic)
            restorer = tf.train.Saver(variables_to_restore)
            restorer.restore(self.sess, "/home/lhc/Desktop/pyECO-master/data/vgg16.ckpt")
            print("loaded")
            self.fix_variables(self.sess, "/home/lhc/Desktop/pyECO-master/data/vgg16.ckpt")
            print("fixed")

    def build_head(self, is_training = False):
        '''建立vgg16网络,返回计算特征需要的avg_net和net
        仅使用vgg16的前四层,最终输出面积为1/4的特征avg_net和1/16的特征net'''
        with tf.variable_scope('vgg_16', 'vgg_16'):
            net = slim.repeat(self._image, 2, slim.conv2d, 64, [3,3], trainable=is_training, scope = 'conv1')
            net = slim.max_pool2d(net, [2,2], padding='SAME', scope='pool1')
            avg_net = slim.avg_pool2d(net, [2,2], padding='SAME', scope='avg_pool1')
            # conv1,1/2

            net = slim.repeat(net, 2, slim.conv2d, 128, [3,3], trainable=is_training, scope='conv2')
            net = slim.max_pool2d(net, [2,2], padding='SAME', scope='pool2')
            # conv2,1/4

            net = slim.repeat(net, 3, slim.conv2d, 256, [3,3], trainable=is_training, scope='conv3')
            net = slim.max_pool2d(net, [2,2], padding='SAME', scope='pool3')
            #conv3,1/8

            net = slim.repeat(net, 3, slim.conv2d, 512, [3,3], trainable=is_training, scope='conv4')
            net = slim.max_pool2d(net, [2,2], padding='SAME', scope='pool4')
            #conv4,1/16

        return avg_net, net

    def get_variables_to_restore(self, variables, var_keep_dic):
        '''从新建模型中获得各层参数名称,以便读取预训练模型中的参数,其中conv1的第一层网络需要调整RGB通道'''
        variables_to_restore = []
        self.variables_to_fix = {}
        for v in variables:
            if v.name == 'vgg_16/conv1/conv1_1/weights:0':
                self.variables_to_fix[v.name] = v
                continue
            if v.name.split(':')[0] in var_keep_dic:
                print('Variables restored : %s' % v.name)
                variables_to_restore.append(v)
        return variables_to_restore

    def fix_variables(self, sess, file_name):
        '''调整conv1的第一层网络权重'''
        print('Fixed VGG16...')
        with tf.variable_scope('Fixed_VGG16'):
            with tf.device("/cpu:0"):
                conv1_rgb = tf.get_variable("conv1_rgb", [3,3,3,64], trainable=False)
                restorer_rgb = tf.train.Saver({"vgg_16/conv1/conv1_1/weights":conv1_rgb})
                restorer_rgb.restore(sess, file_name)
                sess.run(tf.assign(self.variables_to_fix['vgg_16/conv1/conv1_1/weights:0'], tf.reverse(conv1_rgb, [2])))

    def get_variables_in_checkpoint(self, file_name):
        '''读取ckpt文件中的预训练权重'''
        try:
            reader = pywrap_tensorflow.NewCheckpointReader(file_name)
            var_to_shape_map = reader.get_variable_to_shape_map()
            return var_to_shape_map
        except Exception as e:
            print(str(e))
            if "corrupted compressed block contents" in str(e):
                print("Wrong checkpoint file")

