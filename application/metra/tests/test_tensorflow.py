import tensorflow as tf

# 检查 TensorFlow 版本
print("TensorFlow version:", tf.__version__)

# 创建一个简单的 TensorFlow 常量
hello = tf.constant('Hello, TensorFlow!')

# 创建一个 TensorFlow 会话
tf.print(hello)

print(f'\nTensorflow version = {tf.__version__}\n')
print(f'\n{tf.config.list_physical_devices("GPU")}\n')