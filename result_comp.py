import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    font = {'family': 'Times New Roman',
            'style': 'normal',
            'weight': 'normal',
            'color': 'black',
            'size': 12
            }
    plt.rc('font', family='Times New Roman')
    plt.figure()
    result0_1 = pd.read_csv('results3.0.csv')
    data0_1 = result0_1['psnr']
    curve0_1 = plt.plot(data0_1,label='Original')

    result1_1 = pd.read_csv('results3.2.csv')
    data1_1 = result1_1['psnr']
    curve1_1 = plt.plot(data1_1, label='Add Cat')

    result1_3 = pd.read_csv('results3.1.csv')
    data1_3 = result1_3['psnr']
    curve1_3 = plt.plot(data1_3, label='Add cross space')

    result1_2 = pd.read_csv('results3.3.csv')
    data1_2 = result1_2['psnr']
    curve1_2 = plt.plot(data1_2, label='Add long residual1')

    # result1_4 = pd.read_csv('results1.4.csv')
    # data1_4 = result1_4['psnr']
    # curve1_4 = plt.plot(data1_4, label='Add long residual12')

    plt.xlabel('Epoch', fontdict=font)
    plt.ylabel('PSNR', fontdict=font)
    plt.legend(loc='lower right')
    plt.show()
