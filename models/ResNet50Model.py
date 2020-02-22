import tensorflow as tf
DEFAULT_PADDING = 'SAME'
branchNdx = [chr(i) for i in range(97,123)]

def interleave(tensors, axis):
    old_shape = tensors[0].get_shape().as_list()[1:]
    new_shape = [-1] + old_shape
    new_shape[axis] *= len(tensors)
    return tf.reshape(tf.stack(tensors, axis + 1), new_shape)

class ResNet50UpProj(object):
    def __init__(self, input, seed, reuse=False, trainable=False):
        self.input = input
        # self.batch = batch
        self.reuse = reuse
        self.trainable = trainable
        self.seed = seed

    def make_variable(self, name, shape,initializer,trainable):
        return tf.get_variable(name,
                               shape,
                               dtype=tf.float32,
                               initializer=initializer,
                               trainable=trainable)

    def conv(self,
             input,
             c_o,
             name,
             ks=[3,3],
             stride=1,
             biased=False,
             padding=DEFAULT_PADDING):
        c_i = input.get_shape()[-1]
        convolve = lambda i, k: tf.nn.conv2d(i,k,[1,stride,stride,1], padding=padding)
        with tf.variable_scope(name, reuse=self.reuse) as scope:
            kernel = self.make_variable(name='weights',
                                        shape=[ks[0], ks[1], c_i, c_o],
                                        initializer=tf.truncated_normal_initializer(0.0,0.01,seed=self.seed,dtype=tf.float32),
                                        trainable=self.trainable)
            output = convolve(input,kernel)
            if biased:
                biases = self.make_variable('biases',
                                            [c_o],
                                            initializer=tf.constant_initializer(0.0),
                                            trainable=self.trainable)
                output = tf.nn.bias_add(output,biases)

            return output

    def max_pool(self,
                 input,
                 ks,
                 stride,
                 name,
                 padding=DEFAULT_PADDING):
        return tf.nn.max_pool(input,
                              ksize=[1,ks[0],ks[1],1],
                              strides=[1,stride,stride,1],
                              padding=padding,
                              name=name)

    def batch_normalization(self,
                           input,
                           name,
                           scale_offset=True):
        shape = [input.get_shape()[-1]]
        epsilon = 1.e-4
        decay = 0.999
        with tf.variable_scope(name,reuse=self.reuse) as scope:
            pop_mean = self.make_variable('mean',
                                           shape,
                                           initializer=tf.constant_initializer(0.0),
                                           trainable=False)
            pop_var = self.make_variable('variance',
                                         shape,
                                         initializer=tf.constant_initializer(1.0),
                                         trainable=False)
            if scale_offset:
                scale = self.make_variable('scale',
                                           shape,
                                           initializer=tf.constant_initializer(1.0),
                                           trainable=self.trainable)
                offset = self.make_variable('offset',
                                            shape,
                                            initializer=tf.constant_initializer(0.0),
                                            trainable=self.trainable)
            else:
                scale, offset = (None, None)

            if self.trainable:
                batch_mean, batch_var = tf.nn.moments(input, [0,1,2])
                train_mean = tf.assign(pop_mean,
                                       pop_mean * decay + batch_mean * (1 - decay))
                train_var = tf.assign(pop_var,
                                      pop_var * decay + batch_var * (1 - decay))
                with tf.control_dependencies([train_mean,train_var]):
                    output = tf.nn.batch_normalization(input,
                                                       batch_mean,
                                                       batch_var,
                                                       offset,
                                                       scale,
                                                       epsilon,
                                                       name=name)
            else:
                output = tf.nn.batch_normalization(input,
                                                   pop_mean,
                                                   pop_var,
                                                   offset,
                                                   scale,
                                                   epsilon,
                                                   name=name)

            return output

    def conv_bn_relu_block(self,
                        input,
                        c_o,
                        conv_name,
                        bn_name,
                        ks,
                        stride,
                        relu=False):

        output = self.conv(input,
                           c_o,
                           conv_name,#??????
                           ks=ks,
                           stride=stride,
                           biased=False,
                           padding=DEFAULT_PADDING)
        output = self.batch_normalization(output,
                                          bn_name,
                                          scale_offset=True)
        if relu:
            output = tf.nn.relu(output)

        return output

    def bottleneck(self,
                   input,
                   channels,
                   name_prefix,
                   stride):
        output=input
        for i in range(3):
            c_o = channels//4 if i<2 else channels
            ks = [3,3] if i==1 else [1,1]
            relu = True if i<2 else False
            if i==0:
                stride = stride
            else:
                stride = 1
            conv_name = name_prefix + branchNdx[i]
            bn_name = conv_name.replace('res', 'bn')
            output =  self.conv_bn_relu_block(output,
                                         c_o=c_o,
                                         conv_name=conv_name,
                                         bn_name=bn_name,
                                         ks=ks,
                                         stride=stride,
                                         relu=relu)
        return output

    def resnet50_v1_model(self,input):
        # head conv
        head = self.conv_bn_relu_block(input,
                                    c_o=64,
                                    conv_name='conv1',
                                    bn_name='bn_conv1',
                                    ks=[7,7],
                                    stride=2,
                                    relu=False)
        pool1 = self.max_pool(head,
                              ks=[3,3],
                              stride=2,
                              name='pool1')
        rep_dict={'stage2':3,
                        'stage3':4,
                        'stage4':6,
                        'stage5':3}
        feat_dict = {'stage2': 256,
                          'stage3': 512,
                          'stage4': 1024,
                          'stage5': 2048}
        output = pool1
        for stage in range(2,6):
            stage_name = 'stage' + str(stage)
            conv_name = 'res' + str(stage) + 'a' + '_branch1'
            bn_name = conv_name.replace('res','bn')
            name_prefix = 'res' + str(stage) + 'a' + '_branch2'
            channels = feat_dict[stage_name]
            stride = 2 if stage>2 else 1
            br1 = self.conv_bn_relu_block(output,
                                          c_o=channels,
                                          conv_name=conv_name,
                                          bn_name=bn_name,
                                          ks=[1,1],
                                          stride=stride,
                                          relu=False)
            br2 = self.bottleneck(output,
                                  channels=channels,
                                  name_prefix=name_prefix,
                                  stride=stride)
            output = tf.nn.relu(tf.add(br1,br2),
                                name='res'+ str(stage) + 'a_relu')
            for i in range(1,rep_dict['stage'+str(stage)]):
                name_prefix = 'res' + str(stage) + branchNdx[i] + '_branch2'
                bottleneck = self.bottleneck(output,
                                             channels=channels,
                                             name_prefix=name_prefix,
                                             stride=1)
                output = tf.nn.relu(tf.add(output, bottleneck),
                                    name='res' + str(stage) + branchNdx[i] + '_relu')

        return output

    def unpool_as_conv(self,
                       input,
                       c_o,
                       id,
                       relu=False):
        #convolution A(3x3)
        layername = 'layer%s_ConvA'%(id)
        outputA = self.conv(input,
                            c_o,
                            name=layername,
                            ks=[3, 3],
                            stride=1,
                            biased=False,
                            padding='SAME')
        #convolution B(2x3)
        layername = 'layer%s_ConvB'%(id)
        padded_input_B = tf.pad(input, [[0,0],[1,0],[1,1],[0,0]],"CONSTANT")
        outputB = self.conv(padded_input_B,
                            c_o,
                            name=layername,
                            ks=[2,3],
                            stride=1,
                            biased=False,
                            padding='VALID')
        #convolution C(3x2)
        layername = 'layer%s_ConvC'%(id)
        padded_input_C = tf.pad(input, [[0, 0], [1, 1], [1, 0], [0, 0]], "CONSTANT")
        outputC = self.conv(padded_input_C,
                            c_o,
                            name=layername,
                            ks=[3, 2],
                            stride=1,
                            biased=False,
                            padding='VALID')
        # convolution D(2x2)
        layername = 'layer%s_ConvD' % (id)
        padded_input_D = tf.pad(input, [[0, 0], [1, 0], [1, 0], [0, 0]], "CONSTANT")
        outputD = self.conv(padded_input_D,
                            c_o,
                            name=layername,
                            ks=[2, 2],
                            stride=1,
                            biased=False,
                            padding='VALID')

        # Interleaving elements of the four feature maps
        # --------------------------------------------------
        left = interleave([outputA, outputB], axis=1)  # columns
        right = interleave([outputC, outputD], axis=1)  # columns
        output = interleave([left, right], axis=2)  # rows

        #BN
        layername = 'layer%s_BN'%(id)
        output = self.batch_normalization(output,name=layername,scale_offset=True)
        if relu:
            output = tf.nn.relu(output)

        return output

    def up_project(self,
                   input,
                   c_o,
                   id):
        #branch 1
        id_br1 = '%s_br1'%(id)
        output_br1 = self.unpool_as_conv(input,c_o,id_br1,relu=True)
        # Convolution following the upProjection on the 1st branch
        conv_name = "layer%s_Conv" % (id)
        bn_name = 'layer%s_BN' % (id)
        # conv+bn
        output_br1 = self.conv_bn_relu_block(output_br1,
                                             c_o,
                                             conv_name,
                                             bn_name,
                                             ks=[3,3],
                                             stride=1,
                                             relu=False)

        #branch 2
        id_br2 = "%s_br2" % (id)
        output_br2 = self.unpool_as_conv(input,c_o,id_br2,relu=False)

        layerName = "layer%s_ReLU" % (id)
        output = tf.nn.relu(tf.add(output_br1,output_br2), name=layerName)

        return output

    def build_model(self):
        input = self.input
        res_feat = self.resnet50_v1_model(input) # 2048 channels
        output = self.conv_bn_relu_block(res_feat,
                                         c_o=1024,
                                         conv_name='layer1',
                                         bn_name='layer1_BN',
                                         ks=[1,1],
                                         stride=1,
                                         relu=True)
        output = self.up_project(output,512,id='2x')
        output = self.up_project(output,256,id='4x')
        output = self.up_project(output,128,id='8x')
        output = self.up_project(output,64,id='16x')
        output = self.conv(output,c_o=1,
                           name='ConvPred',
                           ks=[3,3],stride=1,
                           biased=True,padding='SAME') #???'VALID'

        return output