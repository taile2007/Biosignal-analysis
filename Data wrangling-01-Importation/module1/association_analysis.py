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
import scipy.stats as stats
from scipy.stats import f as f_test, shapiro as sw
from scipy import special
from scipy.ndimage import uniform_filter
from scipy.spatial.distance import cdist

import pickle

from statsmodels.tsa.seasonal import seasonal_decompose

from numba import njit
from statsmodels.stats.multitest import multipletests

from sklearn.preprocessing import MinMaxScaler
from tslearn import metrics

from itertools import product, combinations

# Lọc dữ liệu

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

def dc_filter(sig,fs):

    '''Hàm thực hiện lọc dữ liệu ở tần số 0.1 - 1.5
    '''

    input = sig.tolist()
    lowcut = 0.1
    highcut = 1.5
    y_filtered, a, b = butter_bandpass_filter(input, lowcut, highcut, fs, order=6)
    sig = y_filtered
    return sig

# Class chuẩn bị dữ liệu cho thí nghiệm

class Assoc_experiment():

    '''Class chuẩn bị dữ liệu cho thí nghiệm Tương quan giữa 2 chuỗi
    '''

    def __init__(self, patient_id: str):

        '''Khởi tạo một database từ dữ liệu đã chọn mẫu và dán nhãn ở định dạng pickle
        # Args:
        # patient_id: tên của gói dữ liệu (bệnh nhân), trong bài này: 'Respiratory'
        
        Tải nội dung database từ file pickle, sau đó lưu lại thành thuộc tính db,
        Ngoài ra tạo các thuộc tính:
        id = tên gói dữ liệu (bệnh nhân);
        resp_code = bảng mã label cho biến cố hô hấp
        '''

        cur_path = os.getcwd()
        df_folder = os.path.join(cur_path, 'D_' + patient_id.split()[0])
        db_name = os.path.join(df_folder, 'DB_' + patient_id.split()[0] + '.pickle')

        with open(db_name, 'rb') as handle:
            db = pickle.load(handle)
        
        self.db = db
        self.id = patient_id
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
        # Output: 1 dataframe gồm 4 cột, tương ứng 4 kênh tín hiệu 
        'Tx_RIP', 'Th_Flow', 'Abd_RIP', 'OEP', mặc định được lọc bởi dc_filter;
        và đồng bộ về datetime index;
        '''

        signals_df = self.db['Raw_signal']
        signals_df.columns = ['Tx_RIP', 'Th_Flow', 'Abd_RIP', 'OEP']
        
        if filter:
            fs = 5
            print(f'Lọc tín hiệu sử dụng dc_filter 0.1-1.5, fs = {fs}')
            signals_df = signals_df.apply(lambda x: dc_filter(sig = x, fs = fs))
        else:
            print('Không sử dụng dc_filter')
        
        return signals_df
    
    def get_exp_data(self):

        '''Hàm chuẩn bị dữ liệu cho thí nghiệm khảo sát tương quan
        Lần lượt thực hiện:
        1) Tải 4 kênh 'Tx_RIP', 'Th_Flow', 'Abd_RIP', 'OEP' vào dataframe 
        signals_df bằng method get_raw_signals()

        2) Tạo dataframe sampl_df chứa danh sách N đoạn tín hiệu dài 10s (100 đơn vị), 
        cho 2 loại biến cố Obstructive và Central

        3) Tạo dataframe comb_df gồm (N x 100) dòng x 4 kênh, 
        tương ứng với N đoạn tín hiệu độ dài 100 xếp chồng lên nhau (bảng dọc)
        Mỗi đoạn đã được xử lý bằng:
        + MinMax scaler
        + Time series decomposistion, chỉ giữ lại Trend
        
        Kết quả xuất ra gồm comb_df và sampl_df
        '''
        
        signals_df = self.get_raw_signals()
                
        sampl_df = self.get_sampling()
        sampl_df.Resp_lab = sampl_df.Resp_lab.astype('int')

        targ = [3,4,5,6]

        sampl_df = sampl_df.query('Resp_lab in @targ & Resp_score >=80')
        sampl_df.reset_index(inplace = True, drop = True)
        
        recoded = {3: 'Obs', 5 : 'Obs',4: 'Cent',6: 'Cent'}
                
        sampl_df.loc[:,'Resp_lab'] = sampl_df['Resp_lab'].map(recoded)
        
        n = sampl_df.shape[0]
        time_stamp = np.hstack([np.arange(100) for i in range(n)])
        idx = np.repeat(np.arange(n), 100)

        comb_df = pd.DataFrame({'t':time_stamp, 'ID': idx})
        final_df = pd.DataFrame(columns= ['Tx_RIP','Th_Flow','Abd_RIP','OEP'])

        for i in range(sampl_df.shape[0]):
            spl_unit = sampl_df.iloc[i]

            idx_start = int(spl_unit.start) - 5
            idx_stop = int(spl_unit.end) + 5
            
            frag_df = signals_df.iloc[idx_start:idx_stop].copy()
            frag_df.reset_index(inplace = True, drop = True)
            
            for ch in ['Tx_RIP','Th_Flow','Abd_RIP','OEP']:
                
                scaler = MinMaxScaler()

                frag_df.loc[:, ch] = scaler.fit_transform(frag_df[ch].values.reshape(-1,1))
                
                res = seasonal_decompose(frag_df[ch].values, model='additive', period=10)
                frag_df.loc[:,ch] = res.trend
            
            frag_df.dropna(inplace = True, axis = 0)
            frag_df.reset_index(inplace = True, drop = True)
            
            final_df = pd.concat([final_df, frag_df], axis = 0)

        final_df.reset_index(inplace = True, drop = True)

        comb_df = pd.concat([comb_df, final_df], axis = 1)
        
        return comb_df, sampl_df

# Hàm thực hiện Sliding correlation sử dụng numba 

@njit
def sliding_corr_coeff(a,b,amc,bmc,W):
    L = len(a)-W+1
    out00 = np.empty(L)
    for i in range(L):
        out_a = 0
        out_b = 0
        out_D = 0
        for j in range(W):
            d_a = a[i+j]-amc[i]
            d_b = b[i+j]-bmc[i]
            out_D += d_a*d_b
            out_a += d_a**2
            out_b += d_b**2
        out00[i] = out_D/math.sqrt(out_a*out_b)
    return out00

def sliding_corr(a,b,W, adjust = False):
    am = uniform_filter(a.astype(float),W)
    bm = uniform_filter(b.astype(float),W)

    amc = am[W//2:-W//2+1]
    bmc = bm[W//2:-W//2+1]

    coeff = sliding_corr_coeff(a,b,amc,bmc,W)

    ab = W/2 - 1
    pval = 2*special.btdtr(ab, ab, 0.5*(1 - abs(np.float64(coeff))))
    
    if adjust:
        pval = multipletests(pval)[1]
    
    return coeff,pval

def rolling_corr(comb_df: pd.DataFrame, sample_idx: int, targets: list, W = 25, plot = True, adjust = False):

    df = comb_df[comb_df['ID'] == sample_idx].iloc[:,2:]
    df.reset_index(inplace = True, drop = True)

    rolling_r,rolling_p = sliding_corr(df[targets[0]].values, df[targets[1]].values, W = W, adjust = adjust)

    if plot:
        f,ax=plt.subplots(3,1,figsize=(13,6),sharex=True)
        df_r = df.rolling(window=W,center=True).mean()[targets]
        df_r.dropna(inplace = True, axis = 0)
        df_r.reset_index(inplace = True, drop = True)
        df_r.plot(ax=ax[0])
        ax[0].set(xlabel='Time',ylabel='Amplitude')
        pd.Series(rolling_r).plot(ax=ax[1], color = '#cf1b39')
        ax[1].set(xlabel='Time',ylabel='Pearson r')
        pd.Series(rolling_p).plot(ax=ax[2], color = '#18a8b8')
        ax[2].set(xlabel='Time',ylabel='p values')
        ax[2].hlines(y = 0.05, xmin = 0, xmax = df_r.shape[0], color = 'red', linestyle = 'dashed')
        ax[0].set(title = f"Mẫu số {sample_idx}")

    return rolling_r,rolling_p

def multiple_rolling_corr(comb_df: pd.DataFrame, sampl_df: pd.DataFrame, targets:list, evt_type = 'Obs', W = 25, plot = True):
    res = pd.DataFrame(index = sampl_df.index, columns=[f'r_{j}' for j in range(100-W+1)], dtype = 'float64')

    for i in range(sampl_df.shape[0]):
        df = comb_df[comb_df['ID'] == i].iloc[:,2:]
        df.reset_index(inplace = True, drop = True)
        
        rolling_r,rolling_p = sliding_corr(df[targets[0]].values, df[targets[1]].values, W = W)
        res.iloc[i] = rolling_r
    
    res['Lab'] = sampl_df['Resp_lab']

    if plot:
        f,ax = plt.subplots(figsize=(8,6))
        sns.heatmap(res.iloc[:,:-1][sampl_df['Resp_lab'] == evt_type],cmap='RdBu_r', ax= ax)

    return res

# Hàm thực hiện Lag Cross-correlation 

def crosscorr(df_x, df_y, lag=0):
    """ Hàm thực hiện tương quan trễ giữa 2 chuỗi 
    @Args: 
    - dd_x và df_y: 2 dataframe chứa chuỗi dữ liệu cần phân tích
    - lag : độ trễ, giá trị mặc định = 0
    
    lag : int, default 0
    datax, datay : pandas.Series objects of equal length

    Output: hệ số tương quan (float)
    """
    return df_x.corr(df_y.shift(lag))

def lag_cross_corr(comb_df: pd.DataFrame, sample_idx: int, targ: str, max_lag = 50, plot = True):

    df = comb_df[comb_df['ID'] == sample_idx].iloc[:,2:]
    df.reset_index(inplace = True, drop = True)

    d1 = df['OEP']
    d2 = df[targ]

    rs = [crosscorr(d1,d2, lag) for lag in range(-max_lag,max_lag)]

    if plot:
        offset = np.ceil(len(rs)/2)-np.argmax(rs)
        f,ax=plt.subplots(figsize=(12,3))
        ax.plot(rs)
        ax.axvline(np.ceil(len(rs)/2),color='k',linestyle='--',label='Trung tâm')
        ax.axvline(np.argmax(rs),color='r',linestyle='--',label='Đỉnh đồng bộ')
        ax.set(title=f"Mẫu số {sample_idx}, Offset = {offset} \nOEP<>{targ}", ylim=[-1,1],xlim=[0,len(rs)], xlabel='Offset',ylabel='Pearson r')
        plt.legend()
        plt.show()

    return rs

def multiple_lag_crosscorr(comb_df: pd.DataFrame, sampl_df: pd.DataFrame, targ: str, evt_type = 'Obs', max_lag = 50, plot = True):
    res = pd.DataFrame(index = sampl_df.index, columns=[f'r_{j}' for j in range(100)], dtype = 'float64')

    for i in range(sampl_df.shape[0]):
        df = comb_df[comb_df['ID'] == i].iloc[:,2:]
        df.reset_index(inplace = True, drop = True)
        
        d1 = df['OEP']
        d2 = df[targ]
        rs = [crosscorr(d1,d2, lag) for lag in range(-max_lag,max_lag)]
        
        res.iloc[i] = rs
    
    res['Lab'] = sampl_df['Resp_lab']

    if plot:
        f,ax = plt.subplots(figsize=(8,6))
        
        sns.heatmap(res.iloc[:,:-1][sampl_df['Resp_lab'] == evt_type],cmap='RdBu_r', ax= ax)
        plt.show()

    return res

# DTW similarity test

def DTW_similarity(comb_df: pd.DataFrame, sampl_df: pd.DataFrame):

    sim_df = pd.DataFrame(columns = ['frag','Evt_type','Tx_RIP', 'Th_Flow', 'Abd_RIP'])

    for i in range(sampl_df.shape[0]):
        evt_type = sampl_df.Resp_lab[i]
        
        frag_df = comb_df[comb_df['ID'] == i][['Tx_RIP', 'Th_Flow', 'Abd_RIP', 'OEP']]
        
        sims = [i, evt_type]

        for j in product(frag_df.drop(['OEP'], axis = 1).columns,['OEP']):
            targ = j[0]

            seq1 = frag_df[targ].values.reshape(1,-1,1)
            seq2 = frag_df['OEP'].values.reshape(1,-1,1)

            _, sim = metrics.dtw_path(seq1[0],seq2[0])
            
            sims.append(sim)
    
        sim_df.loc[i] = sims

    return sim_df

def plot_DWT_matrix(comb_df: pd.DataFrame, targ:str, sample_idx : int):

    frag_df = comb_df[comb_df['ID'] == sample_idx]

    seq1 = frag_df[targ].values.reshape(1,-1,1)
    seq2 = frag_df['OEP'].values.reshape(1,-1,1)

    path, sim = metrics.dtw_path(seq1[0], seq2[0])

    sz = frag_df.shape[0]

    plt.figure(1, figsize=(7,6))

    left, bottom = 0.01, 0.1
    w_ts = h_ts = 0.2
    left_h = left + w_ts + 0.02
    width = height = 0.65
    bottom_h = bottom + height + 0.02

    rect_s_y = [left, bottom, w_ts, height]
    rect_gram = [left_h, bottom, width, height]
    rect_s_x = [left_h, bottom_h, width - 0.13, h_ts]

    ax_gram = plt.axes(rect_gram)
    ax_s_x = plt.axes(rect_s_x)
    ax_s_y = plt.axes(rect_s_y)

    mat = cdist(seq1[0], seq2[0])

    sns.heatmap(mat, cmap = 'jet', ax = ax_gram)
    ax_gram.axis("off")
    ax_gram.invert_yaxis()
    ax_gram.autoscale(False)
    ax_gram.plot([j for (i, j) in path], [i for (i, j) in path], "w--",
                linewidth=2.)

    ax_s_x.plot(np.arange(sz), seq2[0], color = '#510385', linewidth=1.8, label = 'OEP')
    ax_s_x.set_xlim((0, sz - 1))
    ax_s_x.axis("off")
    ax_s_x.set(title = f"Mẫu số {sample_idx}")

    ax_s_y.plot(-seq1[0], np.arange(sz), color = '#850318', linewidth=1.8, label = targ)
    ax_s_y.set_ylim((0, sz - 1))
    ax_s_y.axis("off")

    plt.legend()
    plt.show()

# Kiểm định Friedman

class Friedman:
    
    def __init__(self, df, df_format = 'long', var_name = None,
                 subj=None, cond=None, target=None):
        
        if df_format == 'wide':
            df[subj] = df.index
            ldf = pd.melt(df, id_vars = [subj], 
                          value_vars = cond,
                          value_name = target,
                          var_name = var_name
                         )
            ldf[subj]= ldf[subj].astype('object')
            
            wdf = df.drop([subj], axis = 1)
            cond = var_name
                        
        if df_format == 'long':
            wdf = df.pivot(index = subj, 
                          columns = cond,
                          values = target)
            
            ldf = df[[subj, cond, target]]
    
        self.ldf = ldf.dropna(how = 'any', axis = 0)
        self.wdf = wdf.dropna(how = 'any', axis = 0)
        
        self.params = [cond, subj,target]
    
    @classmethod
    def from_formula(cls, df, formula = 'Y ~ C | G'):
        target = formula.split('~')[0].strip()
        cond = formula.split('~')[1].split('|')[0].strip()
        subj = formula.split('~')[1].split('|')[1].strip()
        
        return cls(df = df, 
                   df_format = 'long', 
                   var_name = cond,
                   subj=subj, 
                   cond=cond, 
                   target=target)
            
    def __repr__(self):
        return f"Kiểm định Friedman với k={self.wdf.shape[1]}, n={self.wdf.shape[0]}"
    
    def F_test(self, alpha = 0.05):
        data = self.wdf
        
        stat = pd.DataFrame(index=['Friedman xấp xỉ',
                                   'Friedman chính xác',
                                   'Iman-Davenport'])
        
        X = data.values
        conds = list(data.columns)
        n,k = data.shape
        
        # Tính Friedman Chisquared test
        
        rank_mat = np.zeros(X.shape)
                
        for i in range(n):
            rank_mat[i] = stats.rankdata(X[i, :])
            
        self.rank_mat = rank_mat
        
        # Phương pháp xấp xỉ
        ssb = (rank_mat.sum(axis=0)**2).sum()
        F_approx = (12 / (n * k * (k + 1))) * ssb - 3 * n * (k + 1)
        
        # Phương pháp chính xác
        rj = rank_mat.mean(axis = 0)
        rm = rank_mat.mean()
        SST = n*((rj - rm)**2).sum()
        SSE = ((rank_mat-rm)**2).sum()/(n*(k-1))
        
        F_exact = SST/SSE
        
        # Hiệu chỉnh thứ hạng bằng nhau
        ties = 0
        for i in range(n):
            replist, repnum = stats.find_repeats(X[i])
            for t in repnum:
                ties += t * (t * t - 1)
        c = 1 - ties / float(k * (k * k - 1) * n)
        
        F_approx /= c
        F_exact /= c
        
        dof = (k - 1)
        
        p1 = stats.chi2.sf(F_approx, dof)
        p2 = stats.chi2.sf(F_exact, dof)
        
        # Iman-Davenport F test
        
        Fc = ((n - 1)*F_approx)/(n*(k-1) - F_approx)
        
        dof_1 = k-1
        dof_2 = (k-1)*(n-1)
        
        p3 = f_test.sf(Fc, dof_1, dof_2)
        
        stat['F'] = [F_approx,F_exact,Fc]
        stat['Độ tự do'] = [str(dof), str(dof), str(f'({dof_1},{dof_2})')]
        stat['Giá trị p'] = [p1, p2,p3]
        
        stat['Phủ định H0'] = ['Có thể' if i else 'Không thể' for i in stat['Giá trị p'] < alpha]
        
        return stat
    
    def Nemenyi_test(self):
        df = self.ldf
        cond, subj, target = self.params
        
        groups = df[cond].unique()
        k = groups.size
        n = df[subj].unique().size
        
        df['Rank'] = df.groupby(subj)[target].rank()
        
        R = df.groupby(cond)['Rank'].mean()
        
        vs = np.zeros((k, k))
        combs = combinations(range(k), 2)
        
        def nemenyi_stats(i, j):
            dif = np.abs(R[groups[i]] - R[groups[j]])
            qval = dif / np.sqrt(k * (k + 1.) / (6. * n))
            return qval
        
        tri_upper = np.triu_indices(vs.shape[0], 1)
        tri_lower = np.tril_indices(vs.shape[0],-1)
        vs[:, :] = 0

        for i, j in combs:
            vs[i, j] = nemenyi_stats(i, j)
            
        vs *= np.sqrt(2.)
        vs[tri_upper] = psturng(vs[tri_upper], k, np.inf)
        vs[tri_lower] = vs.T[tri_lower]
        np.fill_diagonal(vs, 1)

        return pd.DataFrame(vs, index=groups, columns=groups)
    
    def Conover_test(self, p_adjust = 'bonferroni'):
        
        df = self.ldf
        cond, subj, target = self.params
        
        groups = df[cond].unique()
        k = groups.size
        n = df[subj].unique().size
        
        df['Rank'] = df.groupby(subj)[target].rank()
        
        R = df.groupby(cond)['Rank'].sum()
        
        def posthoc_conover(i, j):
            dif = np.abs(R.loc[groups[i]] - R.loc[groups[j]])
            tval = dif / np.sqrt(A) / np.sqrt(B)
            pval = 2. * stats.t.sf(np.abs(tval), df=(m*n*k - k - n + 1))
            return pval
        
        A1 = (df['Rank'] ** 2).sum()
        m = 1
        S2 = m/(m*k - 1.) * (A1 - m*k*n*(m*k + 1.)**2./4.)
        T2 = 1. / S2 * (np.sum(R) - n * m * ((m * k + 1.) / 2.)**2.)
        A = S2 * (2. * n * (m * k - 1.)) / (m * n * k - k - n + 1.)
        B = 1. - T2 / (n * (m * k - 1.))
        
        vs = np.zeros((k, k))
        combs = combinations(range(k), 2)

        tri_upper = np.triu_indices(vs.shape[0], 1)
        tri_lower = np.tril_indices(vs.shape[0], -1)
        vs[:, :] = 0
        
        for i, j in combs:
            vs[i, j] = posthoc_conover(i, j)

        if p_adjust:
            vs[tri_upper] = multipletests(vs[tri_upper], method=p_adjust)[1]

        vs[tri_lower] = vs.T[tri_lower]
        np.fill_diagonal(vs, 1)
        
        return pd.DataFrame(vs, index=groups, columns=groups)
    
    def Siegel_Castellan(self, p_adjust = 'bonferroni'):
        
        df = self.ldf
        cond, subj, target = self.params
        
        groups = df[cond].unique()
        k = groups.size
        n = df[subj].unique().size
        
        df['Rank'] = df.groupby(subj)[target].rank()
        
        R = df.groupby(cond)['Rank'].mean()
        
        def posthoc_SC(i, j):
            dif = np.abs(R[groups[i]] - R[groups[j]])
            zval = dif / np.sqrt(k * (k + 1.) / (6. * n))
            return zval
        
        vs = np.zeros((k, k), dtype=np.float)
        combs = combinations(range(k), 2)

        tri_upper = np.triu_indices(vs.shape[0], 1)
        tri_lower = np.tril_indices(vs.shape[0], -1)
        vs[:, :] = 0

        for i, j in combs:
            vs[i, j] = posthoc_SC(i, j)
            
        vs = 2. * stats.norm.sf(np.abs(vs))
        vs[vs > 1] = 1
        
        if p_adjust:
            vs[tri_upper] = multipletests(vs[tri_upper], method=p_adjust)[1]

        vs[tri_lower] = vs.T[tri_lower]
        np.fill_diagonal(vs, 1)

        return pd.DataFrame(vs, index=groups, columns=groups)
    
    def Miller_Friedman(self):
        
        df = self.ldf
        cond, subj, target = self.params
        
        groups = df[cond].unique()
        k = groups.size
        n = df[subj].unique().size
        
        df['Rank'] = df.groupby(subj)[target].rank()
        
        R = df.groupby(cond)['Rank'].mean()
        
        def posthoc_Miller(i, j):
            dif = np.abs(R[groups[i]] - R[groups[j]])
            qval = dif / np.sqrt(k * (k + 1.) / (6. * n))
            return qval
        
        vs = np.zeros((k, k), dtype=np.float)
        combs = combinations(range(k), 2)

        tri_upper = np.triu_indices(vs.shape[0], 1)
        tri_lower = np.tril_indices(vs.shape[0], -1)
        vs[:, :] = 0

        for i, j in combs:
            vs[i, j] = posthoc_Miller(i, j)

        vs = vs ** 2
        vs = stats.chi2.sf(vs, k - 1)
        
        vs[tri_lower] = vs.T[tri_lower]
        np.fill_diagonal(vs, 1)

        return pd.DataFrame(vs, index=groups, columns=groups)