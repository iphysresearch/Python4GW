#! usr/bin/python
#coding=utf-8

# importing the basic library
from __future__ import print_function

from data_utils import *
import time




import warnings
warnings.filterwarnings("ignore")

def GenGW(masses,  T = 1, fs = 8192, peak_time = 0.8, time_shuffle = [0], f_min = 20, temp_file = 'test.dat', H1_file = 'H1.dat', L1_file = 'L1.dat'):

    """
    生成引力波波形. 
    注意: 要求该函数执行时，Panyi 的程序文件夹相对路径位于: './Panyi_code/Panyi'
         可以通过终端命令 './Panyi_code/Panyi --help' 来检查.

    Input:
    - masses: A list. 其中的元素是双黑洞质量对组成的元组. Eg: [(5,5),(10,10),(15,15)]. 需配合 Distribution_of_masses 函数.
    - PREFIX: detector prefix (e.g., 'H1', 'L1', 'V1')
    - T: 信号波形的时长[s]. 默认为1s.
    - fs: 信号的采样率[Hz]. 默认为8192Hz. (注: N = T * fs, N 必须能被2整除)
    - peak_time: 自起始位置起, 波形信号最大峰值所对应的时间[s]. 默认为0.8s. 主要搭配 time_shuffle 参数使用.
    - time_shuffle: A list. 默认为[0], 其中的列表元素表示偏离 peak_time 的时间差[s]. 
                    (生成波形的次序依据偏离时间差的绝对值有小到大顺序)
    - f_min: Int. 默认值为20[Hz]. 表示从 Panyi 程序中生成GW波形的最小频率.(Lower frequency to start waveform in Hz)
             (若因 f_min 过大使得处理后的信号不够 N, 会 f_min-=5 重新循环生成波形, 直到达到需求)
    - temp_file: Str. 默认值为'test.dat'. 该函数运行过程中会在本地目录中缓存的文件名(最终会自动清理删除)
    
    Output:
    - data: DataFrame. GW 波形信号构成的表格, index 是 GW 波形对应的双黑洞质量.
    """
    N = int(T * fs)
    time_shuffle.sort(key = abs)
    
    assert N % 2 ==0 and fs % 2 == 0
    print('({t:s})'.format(t=time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))))
    
    # 初始化
    index = 0
    index_timeshuffle = 0
    fmin = f_min
    data_H1 = pd.DataFrame(np.array([]))
    data_L1 = pd.DataFrame(np.array([]))
    peaktime = peak_time + time_shuffle[index_timeshuffle]
    peakpoint = int(peaktime * N)
    
    while True:
        try:
            m1, m2 = masses[index]
        except IndexError: # 全部 masses 穷尽后退出循环
            break
            
        # 每个信号生成前的时间戳和进度提示
        stamp = '({t:s}) Working on masses=({m1:.2f}|{m2:.2f}) with fmin={fmin:d} (complete percent: {percent:.2f}/100)'.format(t=time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())), m1=m1, m2=m2,fmin=fmin, percent=1.0 * index / len(masses) * 100)
        print(stamp, end='')

        try:
            os.system('./Panyi_code/Panyi --m1 {m1:.2f} --m2 {m2:.2f} --sample-rate {fs:d} --f-min {fmin:d} --outname {file:s}'.format(m1=m1, m2=m2, fs=fs, fmin=fmin, file=temp_file))
            os.system('./detector_strain/MYdetector_strain -D {PREFIX:s} -a 1:23:45 -d 45.0 -p 30.0 -t 1000000000 < {file:s} > {file0:s}'.format(PREFIX='H1', file=temp_file, file0=H1_file))
            os.system('./detector_strain/MYdetector_strain -D {PREFIX:s} -a 1:23:45 -d 45.0 -p 30.0 -t 1000000000 < {file:s} > {file0:s}'.format(PREFIX='L1', file=temp_file, file0=L1_file))
            assert np.loadtxt(H1_file).shape[0] != 0
            assert np.loadtxt(L1_file).shape[0] != 0
        except AssertionError: # 避免 XLALSimDetectorStrainREAL8TimeSeries(): error: input series too long
            fmin -= 5
            continue
            
        try:
            temp = pd.DataFrame(np.loadtxt(H1_file))
            temp0 = pd.DataFrame(np.loadtxt(L1_file))
        except FileNotFoundError as e:
            print(e)

        while True: 
            # For H1
            # 固定最大峰值位置为第peakpoint个采样点
            posmax = temp[1].argmax()
            part1 = temp[1][posmax-peakpoint+1:]
            part1 = pd.Series(part1.values, index=range(1, len(part1)+1))
            lenpart1 = len(part1)
            if lenpart1 < N:
                lenpart2 = N - part1.shape[0]
                part2 = pd.Series(list([part1.values[-1]]) *lenpart2, index=range(lenpart1+1, lenpart1+1+lenpart2))
                temp_H1 = part1.append(part2, ignore_index = False, verify_integrity = True)
            else:
                temp_H1 = part1[:N]
            
            try:
                # 查验
                assert temp_H1.argmax() == peakpoint   # 如果test的max不够peakpoint, 终止载入，做好标记~
            except AssertionError: # Panyi给出并处理后的temp_有可能不够N个
                print()
                print('The number of GW_sample ({m1:.2f}|{m2:.2f}) with fmin={fmin:d} peaktime={peaktime:.2f}s cannot be {N:d}!'.format(m1=m1,m2=m2,fmin=fmin, peaktime=peaktime,N=N))
                fmin -= 5
                break
            
            # For L1
            part1 = temp0[1][posmax-peakpoint+1:]
            part1 = pd.Series(part1.values, index=range(1, len(part1)+1))
            lenpart1 = len(part1)
            if lenpart1 < N:
                lenpart2 = N - part1.shape[0]
                part2 = pd.Series(list([part1.values[-1]]) *lenpart2, index=range(lenpart1+1, lenpart1+1+lenpart2))
                temp_L1 = part1.append(part2, ignore_index = False, verify_integrity = True)
            else:
                temp_L1 = part1[:N]
            
           

            # 删除本地备份文件
            os.system('rm {file:s}'.format(file=temp_file))
            os.system('rm {file:s}'.format(file=H1_file))
            os.system('rm {file:s}'.format(file=L1_file))
            
            # 保存
            data_H1 = pd.concat([data_H1, pd.DataFrame([temp_H1.values], index=['{m1:.2f}|{m2:.2f}'.format(m1=m1,m2=m2)])])
            data_L1 = pd.concat([data_L1, pd.DataFrame([temp_L1.values], index=['{m1:.2f}|{m2:.2f}'.format(m1=m1,m2=m2)])])
            sys.stdout.write("\r")
            try:
                # 更新 peakpoint
                index_timeshuffle += 1
                assert len(time_shuffle) > index_timeshuffle
            except AssertionError: # timeshuffle没有值可用的时候结束循环跳出
                index_timeshuffle = 0
                index += 1    
                fmin = f_min
                break
            finally:
                peaktime = peak_time + time_shuffle[index_timeshuffle]
                peakpoint = int(peaktime * N)
    print('Finished!')
    return data_H1, data_L1


if __name__ == '__main__':
    masses = Distribution_of_masses(mass1_scope = (4,80), mass_step = 1, ratio_scope = (0.1,1), ratio_step = 0.05)
    data = GenGW(masses[:2] ,peak_time = 0.9, time_shuffle = [0])
