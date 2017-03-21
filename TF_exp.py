import tensorflow as tf

sess = tf.Session()

class WeightBias:
    def __init__(self,in_len,out_len):
        self.W = tf.Variable(tf.random_uniform([out_len,in_len]))
        self.b = tf.Variable(tf.random_uniform([out_len]))
    def get_output(self,in_vec):
        return self.b + tf.matmul(in_vec,self.W)

wb = WeightBias(10,5)
invec = tf.random_uniform([10])
out = wb.get_output(invec)

tf.global_variables_initializer()
# size = 100
# randmat1 = tf.random_uniform([size, size], 0, 1)
# randmat2 = tf.random_uniform([size, size], 0, 1)
#
# sum = tf.zeros([size,size])
# for i in range(10):
#     sum = tf.add(sum,tf.matmul(randmat1,randmat2))

#nodeadd = tf.multiply(tf.add(node1,a),b)
#W = tf.Variable([.3], tf.float32)

test_writer = tf.summary.FileWriter('log',sess.graph)
test_writer.add_run_metadata(tf.RunMetadata(), 'step%d' % i)
print()
print(sess.run(out))
