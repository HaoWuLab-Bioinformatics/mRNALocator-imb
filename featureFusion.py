def feature_merge(feature_list, output):
    import pandas as pd
    data_list = [
        'train',
        'test']
    for data in data_list:
        n = 0
        for fea in feature_list:
            # print(fea)
            n = n + 1
            if n == 1:
                dfx = pd.read_csv(
                    './data/feature/' + data + '/' + fea + '.csv',
                    sep=',',
                    header=0,
                    index_col=0)
                # print((dfx.shape[1]))
            else:
                dfn = pd.read_csv(
                    './data/feature/' + data + '/' + fea + '.csv',
                    sep=',',
                    header=0,
                    index_col=0)
                dfx = pd.concat([dfx, dfn], axis=1)
            dfx1 = pd.DataFrame(dfx)
            dfx1.to_csv(
                './data/fusion/' + data + '/' + output,
                sep=",")


def feature_combine():
    feature_list = [
        'TPCP',
        'DACC_k=10',
        'MMI',
        'NMBroto_k=2']
    feature_merge(feature_list, output='set.csv')


if __name__ == '__main__':
    feature_combine()
