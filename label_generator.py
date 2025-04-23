import pandas as pd
import numpy as np

class LabelGenerator():
    def __init__(self,n):
        self.N=n
    def generate(self,df):
        df=np.array(df.loc[:,["Close"]])
        #print(df[self.N-1:].shape)
        num=len(df)-self.N
        #print(num)
        labels=np.ndarray(num)
        for i in range(num):
            if df[self.N-1:][i+1]>df[self.N-1:][i]:
                labels[i]=0     #上昇
            else:
                labels[i]=1     #下降
        return labels

def main():
    df=pd.read_csv("finance_1d.csv")
    train_df=df[:5300]
    test_df=df[5300:]
    gen=LabelGenerator(n=14)
    #print(np.array(test_df.loc[:,["Close"]])[-7:])
    lbl=gen.generate(train_df)
    print(lbl.shape)
    print(np.count_nonzero(lbl))

if __name__ == "__main__":
    main()