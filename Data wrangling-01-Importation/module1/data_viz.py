import os
import sys

import numpy as np
import pandas as pd
import pytz

import matplotlib.pyplot as plt
import matplotlib.dates as md
import seaborn as sns

from scipy.signal import butter, lfilter, freqz, freqs, firwin
import scipy
import math
from scipy import stats

import pickle

from statsmodels.tsa.seasonal import seasonal_decompose

def butter_bandpass(lowcut, highcut, fs, order=5):
    '''Hàm hỗ trợ lọc dữ liệu ở tần số hô hấp
    '''
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a
    
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    '''Hàm hỗ trợ lọc dữ liệu ở tần số hô hấp
    '''
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y,a,b

def data_filter(sig,fs):
    '''Hàm thực hiện lọc dữ liệu ở tần số hô hấp
    '''
    s = sig.tolist()
    lowcut = 0.1
    highcut = 1.5
    y_filtered, a, b = butter_bandpass_filter(data =s, lowcut = lowcut, highcut = highcut, fs = fs, order=5)

    sig = y_filtered
    return sig

class Data_viz():

    '''Class thăm dò trực quan dữ liệu
    '''

    def __init__(self, patient_id: str):

        '''Khởi tạo một database từ dữ liệu đã chọn mẫu và dán nhãn ở định dạng pickle
        # Args:
        # patient_id: tên của gói dữ liệu (bệnh nhân), trong bài này: 'Respiratory'
        
        Tải nội dung database từ file pickle, sau đó lưu lại thành thuộc tính db,
        Ngoài ra tạo các thuộc tính:
        id = tên gói dữ liệu (bệnh nhân);
        hypno_code = bảng mã label cho hypnogram
        resp_code = bảng mã label cho biến cố hô hấp
        '''

        cur_path = os.getcwd()
        df_folder = os.path.join(cur_path, 'D_' + patient_id.split()[0])
        db_name = os.path.join(df_folder, 'DB_' + patient_id.split()[0] + '.pickle')

        with open(db_name, 'rb') as handle:
            db = pickle.load(handle)
        
        self.db = db
        self.id = patient_id
        self.hypno_code = db['Hypno']['Code']
        self.resp_code = db['Resp_evt']['Code']
    
    def __repr__(self):
        return f"Đây là data base cho pack dữ liệu {self.id}"

    def get_sampling(self):
        '''Hàm lấy dataframe về sampling và label
        # output: dataframe, chứa index của các cửa sổ chọn mẫu, 
        # kèm theo label của Resp và Hypno
        '''
        sampl_df = self.db['Sampling']['Label']
        return sampl_df

    def get_raw_signals(self, filter = True):
        '''Hàm trích xuất dữ liệu của 5 kênh tín hiệu
        # Args:
        filter: True = lọc dữ liệu ở tần số hô hấp; False: không dùng filter
        # Output: 1 dataframe gồm 5 cột, tương ứng 5 kênh tín hiệu 
        'Tx_RIP', 'Th_Flow', 'Abd_RIP', 'OEP', 'SpO2', trong đó 4 kênh đầu tiên
        được lọc ở tần số hô hấp; cả 5 kênh được đồng bộ về datetime index;
        '''

        signals_df = self.db['Raw_signal']
        signals_df.columns = ['Tx_RIP', 'Th_Flow', 'Abd_RIP', 'OEP']

        if filter:
            fs = 5
            print(f'Lọc tín hiệu ở tần số hô hấp, fs = {fs}')
            signals_df = signals_df.apply(lambda x: data_filter(sig = x, fs = fs))
        else:
            print('Tải dữ liệu gốc không sử dụng filter')

        signals_df['SpO2'] = self.db['SpO2']

        return signals_df

    def get_resp_evt_list(self):
        '''Hàm trích xuất danh sách biến cố hô hấp'''

        return self.db['Resp_evt']['Evt_list']

    def get_hypno_evt_list(self):
        '''Hàm trích xuất danh sách trạng thái giấc ngủ'''

        return self.db['Hypno']['Evt_list']

    def plot_respir_evt(self, target: int, evt_id: int, pre = 10., post = 10., res_out=False, visible = True):
        
        '''Hàm khảo sát trực quan 5 kênh tín hiệu trong 1 biến cố hô hấp xác định
        
        # Args:
        - target: int, mã số của biến cố (xem bảng mã self.resp_code)
        - evt_id: int, số thứ tự trong danh sách các biến cố cùng loại target
        Thí dụ: target = 3, evt_id = 7 tương tứng với biến cố Obstructive hypopnea (OH) thứ 7
        - pre, post: float; mở rộng khoảng thời gian trước và sau biến cố (tính bằng giây)
        - res_out: True = có xuất ra dataframe dữ liệu; False (mặc định) = chỉ vẽ biểu đồ, không xuất dữ liệu
        - visible: True = Hiển thị biểu đồ; False = Không hiển thị biểu đồ (nếu kèm res_put = True, chỉ xuất dữ liệu)
        
        Hàm sẽ vẽ 5 biểu đồ tuyến kí cho 5 kênh dữ liệu (riêng step_plot cho SpO2)

        # Output: Nếu res_out = True, sẽ xuất ra dataframe dữ liệu trong khoảng thời gian được khảo sát
        '''
        
        targ = [target]

        evt_lst = self.get_resp_evt_list()
        evt_lst = evt_lst.query('evt_value in @targ')
        evt_lst.reset_index(inplace = True, drop = True)

        if evt_lst.shape[0] == 0:
            print(f"Lỗi: Biến cố {target} không tồn tại")
            return None
        else:
            try:
                evt = evt_lst.iloc[[evt_id]]
            except IndexError:
                print(f"Lỗi: Chỉ có {evt_lst.shape[0]} biến cố, hãy dùng evt_id thấp hơn")
                return None
        
        signal_df = self.get_raw_signals(filter = True)

        start_point = (evt['evt_start'].values[0] - pd.Timedelta(f"{pre}s"))
        end_point = (evt['evt_stop'].values[0] + pd.Timedelta(f"{post}s"))

        signal_df = signal_df.loc[start_point:end_point,:]
        signal_df = signal_df.iloc[:,::-1]

        N = signal_df.shape[0]
        t0 = 0
        dt = 0.1
        time = np.arange(0, N) * dt + t0

        plt.rcParams.update({'font.size': 10})

        fig, axs = plt.subplots(nrows=5,ncols=1,
                                sharex=True, 
                                sharey=False,
                                dpi = 100,
                                figsize=(6,8))

        fig.subplots_adjust(hspace=0.2)

        channels = list(signal_df.columns)

        sig_pals = ['#5402b3','#b3026c','#b30222','#e02500','#e06100']

        fig.suptitle(f"Biến cố {target}-thứ {evt_id}")

        for i in range(5):

            seq = signal_df.iloc[:,i].values
            axs[i].set_title(f"Tín hiệu {channels[i]}")

            if i == 0:

                try:
                    base = np.repeat(np.min(seq)-2, N)
                except ValueError:
                    base = np.repeat(85, N)

                axs[0].fill_between(time,
                                    base,
                                    seq, 
                                    step="pre",
                                    color = sig_pals[i],
                                    alpha=0.8)

                axs[0].plot(time,
                            seq,
                            drawstyle="steps", 
                            alpha = 0.9,
                            lw=0.8,
                            color = sig_pals[i])

                axs[0].set_ylabel('SpO2')

            else:
                axs[i].set_ylabel(channels[i])
                axs[i].plot(time, seq, color = sig_pals[i], lw=0.8)

            axs[4].set_xlabel('Thời gian (giây)')

        if visible:
            plt.tight_layout()
            plt.show()

        else:
            plt.close(fig)

        if res_out:
            return signal_df
        else:
            return None

    def decompose_signal(self, target: int, evt_id: int, pre = 10., post = 10.,chan = 'OEP'):

        '''Hàm phân tích chuỗi tín hiệu OEP (áp lực thực quản) 
        trong khoảng thời gian xảy ra biến cố target thứ i, 
        có thể mở rộng 1 khoảng thời gian trước (pre) và sau (post); 
        thành 3 thành phần: chu kì (seasonal), khuynh hướng (trend) và nhiễu (noise)

        #Args:
        - target: int, mã số của biến cố (xem bảng mã self.resp_code)
        - evt_id: int, số thứ tự trong danh sách các biến cố cùng loại target
        Thí dụ: target = 3, evt_id = 7 tương tứng với biến cố Obstructive hypopnea (OH) thứ 7
        - pre, post: float; mở rộng khoảng thời gian trước và sau biến cố (tính bằng giây)
        - chan = tên của kênh tín hiệu: thí dụ 'Tx_RIP', 'Th_Flow', 'Abd_RIP', 'OEP'
        '''

        sig_df = self.plot_respir_evt(target = target, evt_id = evt_id, pre = pre, post = post, res_out=True, visible = False)

        if sig_df is None:
            print('Không thể lấy được tín hiệu cho biến cố này')

        else:
            N = sig_df.shape[0]
            t0 = 0
            dt = 0.1
            time = np.arange(0, N) * dt + t0

            series = sig_df[chan]
            res = seasonal_decompose(series, model='additive', period=10)
            
            components = [res.observed, res.seasonal, res.trend, res.resid]
            comp_names = ['Tín hiệu gốc','Chu kì','Khuynh hướng','Nhiễu']
            sig_pals = ['#5402b3','#b3026c','#b30222','#e02500',]

            plt.rcParams.update({'font.size': 10})

            fig, axs = plt.subplots(nrows=4,ncols=1,
                                    sharex=True, 
                                    sharey=False,
                                    dpi = 100,
                                    figsize=(6,6))

            fig.subplots_adjust(hspace=0.2)

            fig.suptitle(f"Tín hiệu {chan} trong biến cố {target}-thứ {evt_id}")

            for i in range(4):
                axs[i].set_ylabel(comp_names[i])
                axs[i].plot(time, components[i], color = sig_pals[i], lw=0.8)
            
            axs[3].set_xlabel('Thời gian (giây)')
            plt.tight_layout()
            plt.show()
            



    

            

        

        






