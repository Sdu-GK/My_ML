import numpy as np
import random
import matplotlib.pyplot as plt

# 可以根据需要更改x与y的数据。
x_data = [338, 333, 328, 207, 226, 25, 179, 60, 208, 606]
y_data = [640, 633,619, 393, 428, 27, 193, 66, 226, 1591]

# 损失函数，为预测值与实际值的差的平方和
z = 0

# 选择拟合函数次数
h = int(input("Please input the highest power of polynomial(Not greater than 4):"))
while h > 4:
	h = int(input("Please input the highest power of polynomial(Not greater than 4):"))

# 选取四次函数形式：y_data = b + w1*x_data + w2*(x_data)^2 + w3*(x_data)^3 + w4*(x_data)^4
# 初始化偏置
b = 0
# 初始化权重
w1 = 0
w2 = 0
w3 = 0
w4 = 0
# 定义学习率，通过调节不同的lr参数可以获得不同的曲线长度
lr = 1 
# 定义迭代次数
iteration = 100000
# 对b、w定制化的学习率lr
lr_b = 0
lr_w1 = 0
lr_w2 = 0
lr_w3 = 0
lr_w4 = 0

# 根据梯度下降法更新w,b
for i in range(iteration):
    # 求解梯度
    b_grad = 0.0
    w1_grad = 0.0
    w2_grad = 0.0
    w3_grad = 0.0
    w4_grad = 0.0
    for n in range(len(x_data)):
        b_grad = b_grad - 2.0*(y_data[n] - b - w1*x_data[n] - w2*(x_data[n]**2)
		 - w3*(x_data[n]**3) - w4*(x_data[n]**4))*1.0
        w4_grad = w4_grad - 2.0*(y_data[n] - b - w1*x_data[n] - w2*(x_data[n]**2) 
		 - w3*(x_data[n]**3) - w4*(x_data[n]**4))*(x_data[n]**4)
        w3_grad = w3_grad - 2.0*(y_data[n] - b - w1*x_data[n] - w2*(x_data[n]**2) 
		 - w3*(x_data[n]**3) - w4*(x_data[n]**4))*(x_data[n]**3)
        w2_grad = w2_grad - 2.0*(y_data[n] - b - w1*x_data[n] - w2*(x_data[n]**2) 
		 - w3*(x_data[n]**3) - w4*(x_data[n]**4))*(x_data[n]**2)
        w1_grad = w1_grad - 2.0*(y_data[n] - b - w1*x_data[n] - w2*(x_data[n]**2) 
		 - w3*(x_data[n]**3) - w4*(x_data[n]**4))*x_data[n]
       
    # 对b、w定制化的学习率lr
    lr_b = lr_b + b_grad ** 2
    lr_w1 = lr_w1 + w1_grad ** 2
    lr_w2 = lr_w2 + w2_grad ** 2
    lr_w3 = lr_w3 + w3_grad ** 2
    lr_w4 = lr_w4 + w4_grad ** 2

    # 对b、w定制化的学习率lr,采用Adagard
    b = b - lr / np.sqrt(lr_b) * b_grad
    if h > 3:
       w4 = w4 - lr / np.sqrt(lr_w4) * w4_grad
    if h > 2:
       w3 = w3 - lr / np.sqrt(lr_w3) * w3_grad
    if h > 1:
       w2 = w2 - lr / np.sqrt(lr_w2) * w2_grad
    w1 = w1 - lr / np.sqrt(lr_w1) * w1_grad

# 询问是否打印预测值
re_1 = input("If print the forecast value?(y/n)")
# 是
if re_1 == 'y':
   while True:
      x = input("Please input the value of x(x must be a integer，'q' means quit.):") 
      if x == 'q':
         break
      else:
         x = int(x)
         y = b + w1*x + w2*(x**2) + w3*(x**3) + w4*(x**4)
         print(y)

# 询问是否绘制点与拟合曲线
re_2 = input("If plot the function curve?(y/n)")
# 是
if re_2 == 'y':
   x = np.arange(np.min(x_data)-1, np.max(x_data)+1)
   y = b + w1*x + w2*(x**2) + w3*(x**3) + w4*(x**4)
   plt.plot(x, y)
   plt.plot(x_data, y_data, 'o')
   plt.plot()
   plt.xlabel('x axis')
   plt.ylabel('y axis')
   plt.xlim(np.min(x_data)-1, np.max(x_data)+1)
   plt.ylim(np.min(y_data)-1, np.max(y_data)+1)
   plt.show()

