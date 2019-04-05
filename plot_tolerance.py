import matplotlib.pyplot as plt
import numpy as np

acc_interval, acc_number = np.loadtxt('ToleranceOutput.txt', delimiter =',', unpack = True)
plt.plot(acc_interval, acc_number)
plt.scatter(acc_interval, acc_number)
plt.xlabel('Tolerance Interval')
plt.ylabel('Number of Houses in Accuracy Interval')
plt.title('Accuracy of Linear Regression Method')
plt.grid()
plt.show()
