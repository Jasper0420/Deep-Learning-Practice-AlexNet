# Deep-Learning-Practice-AlexNet
存在一个bug,在train.py文件中数据集的路径有问题
！！！！
data_root = os.path.abspath(os.path.join(os.getcwd(), "../..")) 
！！！！
修改成
！！！！
data_root = os.path.abspath(os.path.join(os.getcwd(), "./.")) 
！！！！
