# Module 1 cho project Biosignal analysis
# Tác giả: Lê Ngọc Khả Nhi
# Version: 19/12/2020

from datetime import datetime, timedelta

import os
import sys
import re
import numpy as np
import pandas as pd
import pytz
from itertools import islice, groupby, count
import concurrent.futures as cf
from tqdm.notebook import tqdm

from module1.constants import *

class Paths_Info_Summary():
    """Class cho phép liệt kê danh sách tất cả tên file, 
    loại tín hiệu và đường dẫn cho các gói dữ liệu
    """

    def __init__(self, data_folder):
        """Khởi tạo class InfoSummary, bằng cách khai báo 
        tên thư mục data_folder

        : data_folder: tên thư mục chứa các gói dữ liệu (pack) 
        pack là các thư mục thứ cấp, mỗi pack tương ứng với raw dataset của 1 bệnh nhân
        """
        self.data_folder = data_folder
        
    @staticmethod
    def get_info_from_path(path: str):
        """Trích xuất tên pack dữ liệu và tên file
        :path: đường dẫn của 1 file dữ liệu cụ thể
        :xuất kết quả là 1 pd.Series gồm tên pack và tên file
        """
        split_path = path.split(os.sep)
        pack = split_path[3]
        filename = split_path[-1]
        return pd.Series([pack, filename])
    
    @staticmethod
    def find_file_signal(fullpath: str, pack_folder: str):
        """Hàm có công dụng phân loại cho mỗi file dữ liệu. 
        Có 3 loại dữ liệu chính:

        1) raw data: tất cả file chứa trong thư mục Raw data, mỗi file là 1 kênh tín hiệu
        2) Scoring sequence: dữ liệu chuỗi kết quả đa ký giấc ngủ, gồm có:
           2A) Chuỗi kết quả liên tục : thí dụ hypnogram, spo2
           2B) Chuỗi kết quả không liên tục, thí dụ desaturations, Micro arousals, Respir. events
        
        :fullpath: đường dẫn đầy đủ cho mỗi file,
        :pack_folder: tên của pack dữ liệu (hồ sơ bệnh nhân)
        :return: signal type, là 1 string, cho biết tên viết tắt của loại dữ liệu;
        Ghi chú: từ viết tắt này sẽ được dùng cho các hàm tải dữ liệu chuyên biệt cho mỗi loại

        Loại dữ liệu 2A và 2B được xác nhận dựa vào type_dict, 
        Tất cả dữ liệu loại 1 (raw data) được tạo tự động từ tên file raw data
        """
        # signal_type = 'other'
        fullpath = fullpath.lower()
        file_name = fullpath.split(os.path.sep)[-1]
        folder_name = fullpath.split(os.path.sep)[-2]
        
        type_dict = {'hypnogramme':'hypno',
                     'desaturations':'evt_spo2',
                     'spo2':'raw_spo2',
                     'respiratory events':'respi_evt',
                     'micro arousals':'ma',}
        
        if folder_name == 'scoring sequences':
            signal_type = type_dict[file_name[:-4]]
        else:
            signal_type = file_name[:-4]
        
        return signal_type
            
    def create_signal_summary(self):
        """Hàm tạo ra dataframe tóm tắt thông tin về pack dữ liệu,  nó
        liệt kê tất cả file dữ liệu trong pack, 
        đường dẫn cho mỗi file,
        Tên của pack (cho phép chọn riêng từng pack)
        Tên file, và extension
        Loại dữ liệu
        :return: pandas dataframe
        mỗi hàng là 1 file, và 4 cột "fullpath", "pack", "filename" và "signal type"
        """
        # Lấy danh sách files
        paths = []
        for root, _, files in os.walk(self.data_folder, topdown=False):
            for name in files:
                paths.append(os.path.join(root, name))
        df_paths = pd.DataFrame(paths, columns=['full_paths'])
    
        # Trích xuất thư mục
        cols = list(df_paths.columns)
        cols.extend(["pack", "filename"])
        df_paths[['pack', 'filename']] = df_paths['full_paths'].apply(
            self.get_info_from_path)

        # Lọc bỏ kí tự đặc biệt trong tên file
        df_paths = df_paths[~df_paths['filename'].str.startswith('._')]
        df_paths = df_paths[~df_paths['filename'].str.startswith('.~')]

        # Trích xuất loại dữ liệu
        df_paths['signal_type'] = \
            df_paths[['full_paths', 'pack']].apply(lambda x: self.find_file_signal(x[0], x[1]),
                                                      axis=1)

        # Loại bỏ tất cả những file không phải là dữ liệu
        df_paths = df_paths[df_paths['signal_type'] != 'other']
        return df_paths

def get_signal_path(file_summary: pd.DataFrame, signal_type: str):
    """Lấy đường dẫn đến 1 file dữ liệu cụ thể, từ dataframe Summary,
    :file_summary: object pd.DataFrame chứ thông tin về pack dữ liệu
    :param signal_type: tên viết tắt của loại tín hiệu/chuỗi cần quan tâm
    :return: path (hoặc None và thông báo nếu không tìm thấy file mục tiêu)
    """
    selected_paths = file_summary.loc[file_summary['signal_type']
                                      == signal_type, 'full_paths']
    if selected_paths.shape[0] >= 1:
        raw_data_path = selected_paths.values[0]
        if selected_paths.shape[0] > 1:
            print(f"Có nhiều hơn 1 file cho tín hiệu {signal_type},chỉ dùng file đầu tiên !")
    else:
        print(f"Không tìm thấy data cho tín hiệu  {signal_type} ở bệnh nhân này !")
        raw_data_path = None
    
    return raw_data_path

class DataImporter():
    """Class cho phép tải dữ liệu từ 1 data pack,
    Mỗi instance tương ứng với 1 pack cụ thể (bệnh nhân)
    """

    def __init__(self, data_path, pack_folder, signal_summary):
        """
        Khởi tạo 1 instance dataimporter
        :data_path: tên của thư mục chứa các pack dữ liệu,
        :pack_folder: tên của 1 pack cụ thể (hay 1 bệnh nhân)
        :signal_summary: tên dataframe đã được tạo ra từ class Paths_Info_Summary, 
        mỗi hàng là 1 file, và 4 cột "fullpath", "pack", "filename" và "signal type"
        """
        self.pack_folder = pack_folder
        self.data_path = os.path.join(data_path, self.pack_folder)
        self.signal_summary = signal_summary[signal_summary['pack'] == pack_folder]
        
    def __repr__(self):
        return f"Đây là data paser cho pack dữ liệu {self.pack_folder}"
    
    def Import_pack_profile(self):
        """Hàm truy xuất thông tin bệnh nhân:
        Đọc file Profile.txt và tạo ra 1 dict gồm Giới tính, chiều cao, 
        cân nặng, BMI cho bệnh nhân.
        """
        
        pack_data_path = os.path.join(self.data_path, 'Profile.txt')
        
        patient_dict = {}
        
        try:
            with open(pack_data_path, 'r') as file:
                tmp = file.readlines()
        
            for line_i in tmp:
                val_i = line_i.split(":")[1].strip()
                varname = PATIENT_INFO_DICT[line_i.split(":")[0].strip()]

                try:
                    val_i = float(val_i)
                except ValueError:
                    pass
                patient_dict[varname] = val_i
        
        except FileNotFoundError:
            print(f"Không tìm thấy thông tin bệnh nhân !")
            
        return patient_dict
    
    @staticmethod
    def adjacent_to_irregular(df_resampled: pd.Series, samp_freq : int):

        '''Hàm chuyển đổi từ loại dữ liệu chuỗi 2A (sau tái chọn mẫu), sang loại dữ liệu chuỗi 2B
        Chỉ áp dụng cho chuỗi kết quả Hypnogram
        : df_resample: pandas Series, có nội dung là 1 chuỗi liên tục các label, có datetime index,
        : samp_freq: tần số lấy mẫu, 1 số nguyên, thí dụ 10, 30
        : return: 1 dataframe có cấu trúc: mỗi hàng là 1 biến cố,
        4 cột: evt_start/stop = thời điểm bắt đầu, thời điểm kết thúc;
        evt_info: độ dài của biến cố, tính bằng giây,
        evt_value: Nhãn của biến cố, quy định trong dictionary
        '''

        fs = np.round(float(1/samp_freq),5)

        grp = np.array([[int(k),float(len(list(g))/samp_freq)] for k,g in (groupby(df_resampled))])
            
        change_point = np.array(df_resampled.diff()[lambda x: x != 0].index.tolist())

        df_event = pd.DataFrame({'evt_start': change_point,
                         'evt_stop': np.array([s + pd.Timedelta(f"{i}S") - pd.Timedelta(f"{fs}S") for s,i in zip(change_point, grp[:,1])]),
                          'evt_info': grp[:,1],
                          'evt_value': grp[:,0],
                         })

        return df_event

    def import_raw_data(self, signal_type: str, n_lines=None, samp_freq=None):
        """Hàm cho phép tải bất kì dữ liệu nào thuộc loại 1 (raw data), 
        áp dụng tổng quát cho tất cả raw data
        
        :signal_type: tên viết tắt của loại dữ liệu, chú ý: luôn là chữ thường
        :n_lines: số nguyên, giới hạn số hàng cần tải, thấp nhất là 5;

        Ghi chú: 
        1) Nếu n_lines = None, hàm sẽ xuất ra 3 kết quả:
        :return 1: df_data, tín hiệu gốc, 
                   pd.Series liên tục, có datetime index
        :return 2: df_resampled, tín hiệu tái chọn mẫu; 
                   pd.Series liên tục, có datetime index
        :return 3: fs, giá trị tần số lấy mẫu gốc, int

        2) Nếu n_lines = 5, hàm chỉ xuất ra 2 kết quả:
        :return 1: pd.Series gồm 2 hàng: thới điểm bắt đầu và kết thúc xét nghiệm
        :return 2: fs: giá trị tần số lấy mẫu gốc, int

        3) Nếu không tìm thấy đường dẫn của loại tín hiệu:
        Xuất ra 3 kết quả tương tự (1), nhưng tất cả đều rỗng (None)
        """
        raw_data_path = get_signal_path(self.signal_summary, signal_type)
        
        if raw_data_path is None:
            return None, None, None

        df_data = pd.DataFrame()
        with open(raw_data_path, 'r') as file:
            if n_lines:
                lines = list(islice(file, n_lines))
            else:
                lines = file.readlines()

        # Xác định thời điểm bắt đầu ghi tín hiệu
        try:
            start_datetime = datetime.strptime(lines[1].split('Time:')[1].strip(),
                                               '%d-%m-%y %H:%M:%S')
        except ValueError:
            start_datetime = datetime.strptime(lines[1].split('Time:')[1].strip(),
                                               '%d/%m/%Y %H:%M:%S')
        # Xác định tần số lấy mẫu
        if lines[2].split(":")[0].strip() == "Sample Rate":
            fs = int(lines[2].split(":")[1].strip())
        else:
            fs = None
            return df_data, fs
        
        # Xác định độ dài
        if lines[3].split(":")[0].strip() == "Length":
            length = int(lines[3].split(":")[1].strip())
            stop_datetime = start_datetime + \
                timedelta(seconds=int(length/fs))  # +/- 1?

        # Nếu chỉ cần lấy thời điểm khởi đầu, kết thúc và tần số
        if n_lines == 5:
            df_times = pd.Series(index=[pytz.timezone("Europe/Paris").localize(start_datetime),
                                        pytz.timezone("Europe/Paris").localize(stop_datetime)])
            return df_times, fs

        # Định vị dữ liệu
        if lines[6].strip() == "Data:":
            values = lines[7:]
            values = np.array([float(x.strip().replace(',', '.'))
                               for x in values])
        else:
            return df_data, fs
        time = [pytz.timezone("Europe/Paris").localize(start_datetime + x*timedelta(seconds=1/fs))
                for x in range(len(values))]

        df_data = df_data = pd.Series(values, index=time, name=signal_type)

        # Tái chọn mẫu
        if samp_freq is not None:
            if isinstance(samp_freq, int):
                df_resampled = df_data.resample(f"{np.round(1/samp_freq, 5)}S").bfill()
            else:
                print('giá trị tần số phải là số nguyên')
                df_resampled = None
        else:
            df_resampled = None
        
        return df_data, df_resampled, fs
    
    @staticmethod
    def fill_new_axis(df_data, new_axis):
        """Hàm tái chọn mẫu, áp dụng cho loại tín hiệu 2B (danh sách các biến cố, time scale không liên tục) 
        Cho phép hoán chuyển dữ liệu loại 2B (biến cố không liên tục) thành 2A (chuỗi liên tục), 
        và tự động điền khuyết những khoảng còn trống trong chuỗi này
        
        :df_data: data frame dữ liệu thuộc loại 2B (biến cố không liên tục), mỗi hàng là 1 biến cố,
        và 4 cột: 'evt_start, 'evt_stop' (thời điểm bắt đầu, kết thúc), 
        'evt_info'(thời gian biến cố) và 'evt_value' (nhãn biến cố)
        :new_axis: datetime array dùng để tái chọn mẫu cho chuỗi biến cố 
        :return: dataframe chỉ có 1 cột tên là "value": chuỗi biến cố liên tục (loại 2A), với datetime index
        """
        df_resampled = pd.DataFrame(index=new_axis, columns=['value'])
        for _, row_i in df_data.iterrows():
            nearest_start_idx = \
                pd.Series(
                    (df_resampled.index - row_i['evt_start']).total_seconds()).abs().idxmin()
            nearest_stop_idx = \
                pd.Series(
                    (df_resampled.index - row_i['evt_stop']).total_seconds()).abs().idxmin()
            df_resampled.loc[df_resampled.index[nearest_start_idx]:
                             df_resampled.index[nearest_stop_idx], 'value'] = row_i['evt_value']
        return df_resampled
    
    @staticmethod
    def process_jump_time(time_list, start_datetime):
        """Hàm cho phép tính ra ngày tháng năm khi chỉ có time mà không có date, 
        dùng trong trường hợp chuyển tiếp từ đêm hôm trước sang rạng sáng ngày hôm sau.
         :time_list: list các time stamps
         :start_datetime: Ngày thực hiện xét nghiệm, trích xuất từ file raw data
         :return: list date time
         """
        date_str = '%H:%M:%S,%f'
        jump_time = [datetime.strptime(x, date_str).time() for x in time_list]
        start_date = start_datetime.date()
        datetime_list = \
            [pytz.timezone("Europe/Paris").localize(datetime.combine(start_date, x), is_dst=None)
             for x in jump_time]
            
        # Chuyển từ đêm trước sang sáng hôm sau:
        day_diff = np.array([x.days for x in np.diff(datetime_list)])
        if day_diff.sum() != 0:  # khi biến cố hoàn toàn trước hay sau nửa đêm
            idx_day_jump = np.nonzero(day_diff)[0][0]
            np_days = np.zeros((len(jump_time),))
            np_days[idx_day_jump + 1:] = 1
            datetime_list = datetime_list + timedelta(days=1) * np_days
        elif (day_diff.sum() == 0) and (datetime_list[0].hour < 12):
            # khi biến cố sinh ra sau nửa đêm nhưng xét nghiệm bắt đầu đêm hôm qua
            if start_datetime.hour > 12:
                np_days = np.ones((len(jump_time), ))
                datetime_list = datetime_list + timedelta(days=1) * np_days
        return datetime_list
    
    def import_irregular_window_evt(self, signal_type: str, samp_freq=None, get_true_stop_time=False):

        """Hàm cho phép tải dữ liệu chuỗi biến cố thuộc loại 2B : không liên tục;
        hỗ trợ tái chọn mẫu với tần số tùy chọn

        :signal_type: tên viết tắt của loại dữ liệu, thí dụ 'respi_evt', 'ma' hoặc 'evt_spo2'
        :samp_freq: tần số lấy mẫu hoặc datetime array dùng để tái chọn mẫu
        :get_true_stop_time: Đồng bộ hóa với 1 raw data, 
        giá trị mặc định là False, khi đó chỉ sử dụng timestamps từ chuỗi biến cố,
        nếu True, sẽ dùng timestamps của tín hiệu thô tương ứng, 
        thí dụ respi_evt sẽ dùng Nasal flow Thermistor, desaturation sẽ dùng SpO2, ...
        Cách làm này cho phép tối ưu hóa về đồng bộ giữa chuỗi biến cố và tín hiệu thô.

        :return 1: 1 dataframe (chuỗi loại 2B) có cấu trúc mỗi hàng là 1 biến cố, 
        và 4 cột:'evt_start, 'evt_stop', 'evt_info'(thời gian mỗi biến cố) và 'evt_value' (nhãn kết quả)
        :return: 1 dataframe (chuỗi loại 2A) có cấu trúc chuỗi liên tục, với datetime index, 
        đã được tái chọn mẫu theo tần số tùy chọn.

        :return 2: 1 dictionary cho biết ý nghĩa của các label,
        :return 3: Thời điểm bắt đầu xét nghiệm;
        :return 4: Thời điểm kết thúc xét nghiệm

        Ghi chú: Nếu không tìm thấy đường dẫn, sẽ xuất ra 3 object rỗng (None)
        """
        
        data_path = get_signal_path(self.signal_summary, signal_type)
        
        if data_path is None:
            return None, None, None, None

        stop_datetime = None

        if signal_type == 'respi_evt':
            first_data_line = 5
            dict_map = RESPIRATION_DICT
        elif signal_type == 'ma':
            first_data_line = 5
            dict_map = MICRO_AROUSAL_DICT
        elif signal_type == 'evt_spo2':
            first_data_line = 5
            dict_map = DESATURATION_DICT

        with open(data_path, 'r', encoding='latin1') as f:
            lines = f.readlines()

        # Thời điểm bắt đầu ghi
        try:
            start_datetime = datetime.strptime(lines[1].split('Time:')[1].strip(),
                                               '%d-%m-%y %H:%M:%S')
        except ValueError:  # what is the frequency of this error?
            start_datetime = datetime.strptime(
                lines[1].split('Time:')[1].strip(), '%d-%m-%y')

        start_datetime = pytz.timezone(
            "Europe/Paris").localize(start_datetime, is_dst=None)

        # Thời điểm kết thúc thực sự (dựa vào raw data)
        raw_dict = {'evt_spo2':'spo2',
                    'raw_spo2':'spo2',
                    'respi_evt':'flow th',
                   }
        
        if get_true_stop_time:
            df_start_stop, _  = self.import_raw_data(signal_type = raw_dict[signal_type], n_lines=5)
            
            if df_start_stop is not None:  # raw file exists etc.
                stop_datetime = df_start_stop.index[-1]

        # Dữ liệu chuỗi
        split_str = [x.split(';') for x in lines[first_data_line:]]
        # Không có biến cố nào
        if split_str == []:
            print(f"Không tìm thấy biến cố nào cho {signal_type} trong pack {self.pack_folder}")
            return None, None, None, start_datetime, stop_datetime
        
        # Đọc giá trị chuỗi biến cố
        jump_start_list = [x[0].split('-')[0] for x in split_str]
        jump_end_list = [x[0].split('-')[1] for x in split_str]
        jump_start_datetime = self.process_jump_time(
            jump_start_list, start_datetime)
        jump_end_datetime = self.process_jump_time(
            jump_end_list, start_datetime)
        # Tính thời gian biến cố
        evt_duration = np.array([int(x[1]) for x in split_str])
        evt_type = pd.Series([x[2].strip() for x in split_str])

        # Xuất dataframe
        df_data = pd.DataFrame({'evt_start': jump_start_datetime,
                                'evt_stop': jump_end_datetime,
                                'evt_info': evt_duration,
                                'evt_value': evt_type})

        # Xấp xỉ thời điểm kết thúc
        if stop_datetime is None:
            stop_datetime = df_data['evt_stop'].max()

        # Mapping nhãn và dictionary
        df_data['evt_value'] = df_data['evt_value'].map(dict_map)
        # Kiểm tra xem có nhãn nào bị thiếu không ?
        assert df_data['evt_value'].isna().sum() == 0

        # Tái chọn mẫu
        if samp_freq is not None:
            if isinstance(samp_freq, int):
                new_axis = [start_datetime + x*timedelta(seconds=1/samp_freq)
                            for x in range(int((jump_end_datetime[-1] - start_datetime)
                                               .total_seconds()*samp_freq))]
            else:
                new_axis = samp_freq.copy()
            df_resampled = self.fill_new_axis(df_data, new_axis)
            df_resampled['value'].fillna(value=0, inplace=True)
        else:
            df_resampled = None
        return df_data, df_resampled, dict_map, start_datetime, stop_datetime
    
    def import_adjacent_window_evt(self, signal_type='hypno', samp_freq = None, reshape = True):
        
        """Hàm tải dữ liệu chuỗi biến cố liên tục (loại 2A), thí dụ hypno; 
        hỗ trợ tái chọn mẫu và hoán chuyển ngược về loại 2B (danh sách biến cố không liên tục)

        :signal_type: tên viết tắt loại dữ liệu, Lưu ý: hàm này chỉ dùng cho 'hypno' hoặc 'spo2'
        :reshape = True có nghĩa là sẽ hoán chuyển 2A thành 2B, và xuất ra cả 2; 
        : reshape = False có nghĩa là không cần hoán chuyển, chỉ xuất ra dữ liệu chuỗi liên tục 2A
        :return 1: df_data: 1 pd.Series chứa dữ liệu chuỗi liên tục (gốc), có datetime index
        :return 2: df_resampled: 1 pd.Series chứa dữ liệu chuỗi liên tục (đã được tái chọn mẫu), có datetime index
        :return 3: dict_map: dictionary cho biết ý nghĩa các nhãn biến cố
        :return 4: df_event: chỉ khi reshape = True, dataframe danh sách biến cố không liên tục (2B), 
        có cấu trúc mỗi hàng là 1 biến cố, và 4 cột:'evt_start, 'evt_stop', 'evt_info'(thời gian mỗi biến cố) 
        và 'evt_value' (nhãn kết quả).
        :return 5: window_len: Độ dài của cửa sổ quan sát nếu cố định, = None nếu không cố định
        """
    
        data_path = get_signal_path(self.signal_summary, signal_type)
        
        if data_path is None:
            return None, None, None
        with open(data_path, 'r', encoding='latin1') as file:
            lines = file.readlines()

        dict_map = None

        # Loại hypno: chuỗi labels, có window len
        if signal_type == 'hypno':
            convert_to_numeric = False
            first_data_line = 7
            dict_map = HYPNO_DICT
            need_window_len = True

        # Loại raw_spo2: chuỗi số, không có window length
        elif signal_type == 'raw_spo2':
            convert_to_numeric = True
            first_data_line = 5
            need_window_len = False
        
        # Thời điểm bắt đầu
        try:
            start_datetime = datetime.strptime(lines[1].split('Time:')[1].strip(),
                                               '%d-%m-%y %H:%M:%S')
        except ValueError:  # what is the frequency of this error?
            start_datetime = datetime.strptime(
                lines[1].split('Time:')[1].strip(), '%d-%m-%y')

        split_str = [x.split(';') for x in lines[first_data_line:]]

        # Giá trị thời gian
        jump_start_list = [x[0] for x in split_str]
        jump_datetime = self.process_jump_time(jump_start_list, start_datetime)
        if need_window_len:
            window_len = \
                np.median(np.diff(jump_datetime).astype(
                    "timedelta64[ms]").astype(int) / 1000)
        else:
            window_len = None
        
        # Giá trị
        jump_value = [x[1].strip() for x in split_str]

        # Xuất dataframe
        if convert_to_numeric:
            df_data = pd.Series(pd.to_numeric(jump_value, errors='coerce'), index=jump_datetime,
                                name=signal_type)
        else:
            df_data = pd.Series(jump_value, index=jump_datetime,
                                name=signal_type).map(dict_map)
        
        # Tái chọn mẫu
        if samp_freq is not None:
            if isinstance(samp_freq, int):
                df_resampled = df_data.resample(f"{np.round(1/samp_freq, 5)}S").pad()
            else:
                print('Tần số tái chọn mẫu phải là số nguyên')
        else:
            df_resampled = None
        
        # Hoán chuyển 
        if reshape:
            df_event = self.adjacent_to_irregular(df_resampled, samp_freq)
            return df_data, df_resampled, df_event, dict_map, window_len
        else:
            return df_data, df_resampled, dict_map, window_len
    
    def import_multiple_raw_data(self, type_list: list, samp_freq: int, n_jobs: int):
        '''Hàm cho phép tải đồng thời nhiều kênh tín hiệu thô từ 1 list, 
        và đồng bộ hóa tất cả theo cùng 1 tần số lấy mẫu
        :type_list: 1 list nhiều tên viết tắt của loại tín hiệu;

        Ghi chú: Chỉ áp dụng cho raw data (Loại 1), không áp dụng được cho loại 2A và 2B
        :samp_freq: 1 giá trị tần số lấy mẫu duy nhất, dùng để tái chọn mẫu
        :n_jobs: Số tác vụ thi hành song song (threads)

        :return: 1 dictionary với keys là tên loại tín hiệu, 
        value là pd.Series chứa chuỗi tín hiệu tương ứng, đã được tái chọn mẫu,
        có datetime index.
        '''

        print(f"Khởi động tác vụ song song, tải {len(type_list)} kênh tín hiệu")

        # Dùng lại hàm import_raw_data nhưng chỉ lấy df_resampled
        def raw_parser(signal_type: str, samp_freq: int):
            _, df_resampled, _ = self.import_raw_data(signal_type = signal_type, samp_freq = samp_freq)
            return df_resampled

        # wrapper cho hàm raw_parser, cần cho ThreadPool mapping
        def raw_parser_wrapper(args):
            return raw_parser(*args)

        # Đóng gói args để mapping
        args_generator = ((sig, samp_freq) for sig in type_list)

        # Multithreading
        with cf.ThreadPoolExecutor(max_workers=n_jobs) as p:
            with tqdm(total=len(type_list)) as pbar:
                signal_pack = dict.fromkeys(type_list)

                for i,r in enumerate(p.map(raw_parser_wrapper, args_generator)):
                    pbar.update(1)
                    print(f"Tải và tái chọn mẫu thành công kênh {type_list[i]}")
                    signal_pack[type_list[i]] = r

        print(f"Đã tải xong {len(type_list)} kênh tín hiệu")

        return signal_pack


