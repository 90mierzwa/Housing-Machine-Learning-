import matplotlib.pyplot as plt
import numpy as np

acc_interval, acc_number, acc_reg = np.loadtxt('accuracy_data.txt', delimiter =',', unpack = True)
plt.plot(acc_interval, acc_number)
plt.plot(acc_interval, acc_reg)
# plt.scatter(acc_interval, acc_number)
plt.xlabel('Tolerance Interval')
plt.ylabel('Number of Houses in Accuracy Interval')
plt.title('Comparing Decision Tree and Regression Method')
plt.grid()
plt.show()
