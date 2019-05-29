import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    plt.figure()
    result0_1 = pd.read_csv('results0.1.csv')
    data0_1 = result0_1['psnr']
    curve0_1 = plt.plot(data0_1,label='Original')

    result1_1 = pd.read_csv('results1.1.csv')
    data1_1 = result1_1['psnr']
    curve1_1 = plt.plot(data1_1, label='Add Cat')

    result1_3 = pd.read_csv('results1.3.csv')
    data1_3 = result1_3['psnr']
    curve1_3 = plt.plot(data1_3, label='Add cross space')

    plt.xlabel('Epoch')
    plt.ylabel('PSNR')
    plt.legend(loc='lower right')
    plt.show()
