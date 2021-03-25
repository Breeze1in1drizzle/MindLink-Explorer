import pandas as pd


def one_tsv_2_csv(file_path=None, file_name=None):
    '''
    Description: A (*.tsv) file is going to be converted to (*.csv) one after this function works.
    这里涉及缺失值的处理方法
    原始数据有缺失值，例如 -1、null等等
    (1)整行空白的null，可以drop掉
    (2)如果有-1，则需要进行一定的数据处理

    :param file_path:文件路径
    :param file_name:文件名
    :return:None
    '''
    print("one_tsv_2_csv.start......")
    # 分隔符需要定义，（*.tsv)文件的分隔符是'\t'，而不是默认的' '
    # df = pd.DataFrame(pd.read_csv(file_path + file_name, sep='\t', header=23))
    try:
        df = pd.DataFrame(pd.read_csv(file_name, sep='\t', header=0))
    except:
        print("read_csv error")
        return 0
    # print("df:\n", df)
    # print("df[['Timestamp', 'DateTimeStamp', 'PupilLeft', 'PupilRight']]\n",
    #       df[['Timestamp', 'DateTimeStamp', 'PupilLeft', 'PupilRight']])
    # new_file_name ==>将尾缀的'.tsv'改为'.csv'进行保存
    new_file_name = file_name.replace('.tsv', '.csv')
    try:
        new_df = df[['Timestamp', 'DateTimeStamp', 'PupilLeft', 'PupilRight']]
    except:
        print("new_df error")
        df = pd.DataFrame(pd.read_csv(file_path + file_name, sep='\t', header=17))
        new_df = df[['Timestamp', 'DateTimeStamp', 'PupilLeft', 'PupilRight']]
    # new_df = new_df.drop(df[df['PupilLeft']=='NaN'].index)

    # 去除有空白值的行
    new_df = new_df.dropna(subset=['PupilLeft'])

    # 在原路径进行保存
    new_df.to_csv(file_path + new_file_name, index=False)
    print("one_tsv_2_csv.end......")
    return 1


def drop_the_front_data(file_path, file_name):
    '''
    这段代码与one_tsv_2_csv函数的代码放在一起会有错误，会有index上的识别错误
    猜测是在为保存的时候，虽然有些行被drop掉了，但是index没有更新，需要保存的时候才进行更新
    于是在进行行操作的时候就会有错误
    :param file_path:
    :param file_name:
    :return:
    '''
    new_df = pd.DataFrame(pd.read_csv(file_path+file_name))
    # print(new_df)
    # 去除前一小段不足1s的数据
    time_start = new_df.iat[0, 1]
    # print("time_start:\n", type(time_start), "\n", time_start)
    start_index = 0
    for i in range(len(new_df)):
        time = new_df.iat[i, 1]
        # print("time: ", time)
        str_a = time_start[:8]
        # print("a: ", str_a)
        str_b = time[:8]
        # print("b: ", str_b)
        if str_a == str_b:
            continue
        else:
            start_index = i
            break
    # print("start_index: ", start_index)
    new_df = new_df.drop(labels=range(0, start_index), axis=0)
    new_df.to_csv(file_path+file_name, index=False)


def pupil_diameter_calculation(file_path, file_name):
    '''
    计算瞳孔直径
    :param file_path:
    :param file_name:
    :return:
    '''
    df = pd.DataFrame(pd.read_csv(file_path+file_name))
    # print(df.iat[0, 3])
    # print(df.iat[0, 2])
    # print(df.iat[1, 3])
    # print(df.iat[1, 2])
    pupil_diameter = []
    for i in range(len(df)):
        pupil_diameter.append(abs(df.iat[i, 3] - df.iat[i, 2]))     # -1的时候就变成0（-1往往成对出现）
    df['PupilDiameter'] = pupil_diameter
    df.to_csv(file_path+file_name, index=False)


def raw_data_preprocess(file_path, file_name):
    '''
    :param file_path:
    :param file_name:
    :return:
    '''
    print("raw_data_preprocess.start......")
    # print(file_path+file_name)
    judge = one_tsv_2_csv(file_path='', file_name='P10-Rec1-All-Data-New_Section_30-test.tsv')  # 格式转换+清楚了空白值
    if judge == 0:
        return 0        # 如果有问题，直接跳过该文件夹
    # 去除了最开头不足1s的数据，因为要满足119s的时长，这多出的不足1s的数据无法纳入时间范围，而且这些数据不够后面的稳定
    file_name = file_name.replace('.tsv', '.csv')
    print(file_name)
    drop_the_front_data(file_path=file_path, file_name=file_name)
    pupil_diameter_calculation(file_path=file_path, file_name=file_name)
    print("raw_data_preprocess.end......")


if __name__ == "__main__":
    raw_data_preprocess(file_path='', file_name='P10-Rec1-All-Data-New_Section_30-test.tsv')
