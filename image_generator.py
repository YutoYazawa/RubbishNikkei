import cv2
import numpy as np
import pandas as pd

BLUE=(255,0,0)
RED=(0,0,255)
SIZE=56
N=14

class PriceOnly():
    def __init__(self,n):
        self.N=n
    def generate(self,df):
        df=df.loc[:,["Close","High","Low","Open"]]
        # df=df.loc[:,["Close","High","Low","Open","Volume"]]
        # for Price&Volume Class
        num=len(df)-self.N
        imgs = np.ndarray((SIZE, SIZE, 3, num))
        for i in range(num):
            img = np.full((SIZE, SIZE, 3), 0, dtype=np.uint8)
            sec=df[i:i+self.N]
            sec=(sec-sec.min(None))/(sec.max(None)-sec.min(None))
            for j in range(len(sec)):
                candle_data=sec[j:j+1]
                color=RED
                if candle_data["Close"].item()>candle_data["Open"].item():
                    color=BLUE
                # for Price & Volume Class
                #cv2.rectangle(img, (int(j*(SIZE/self.N)), int((-16+SIZE-40*candle_data["Close"].item())+0.5)), (int((j+1)*(SIZE/self.N)-1), int(-16+SIZE-40*candle_data["Open"].item()+0.5)), color, cv2.FILLED)
                #cv2.line(img, (int(j*(SIZE/self.N))+int(SIZE/self.N/2)-1,int((-16+SIZE-40*candle_data["High"].item())+0.5)),(int(j*(SIZE/self.N))+int(SIZE/self.N/2)-1,int((-16+SIZE-40*candle_data["Low"].item())+0.5)),color)
                cv2.rectangle(img, (int(j*(SIZE/self.N)), int((SIZE-SIZE*candle_data["Close"].item())+0.5)), (int((j+1)*(SIZE/self.N)-1), int(SIZE-SIZE*candle_data["Open"].item()+0.5)), color, cv2.FILLED)
                cv2.line(img, (int(j*(SIZE/self.N))+int(SIZE/self.N/2)-1,int((SIZE-SIZE*candle_data["High"].item())+0.5)),(int(j*(SIZE/self.N))+int(SIZE/self.N/2)-1,int((SIZE-SIZE*candle_data["Low"].item())+0.5)),color)
            imgs[:,:,:,i]=img
        return imgs

class PriceAndVolume():
    def __init__(self,n):
        self.N=n
    def generate(self,df):
        voldf=df.loc[:,["Volume"]]
        df=df.loc[:,["Close","High","Low","Open"]]
        # for Price&Volume Class
        num=len(df)-self.N
        imgs = np.ndarray((SIZE, SIZE, 3, num))
        for i in range(num):
            img = np.full((SIZE, SIZE, 3), 0, dtype=np.uint8)
            sec=df[i:i+self.N]
            volsec=voldf[i:i+self.N]
            sec=(sec-sec.min(None))/(sec.max(None)-sec.min(None))
            volsec=(volsec-volsec.min(None))/(volsec.max(None)-volsec.min(None)+1)
            for j in range(len(sec)):
                candle_data=sec[j:j+1]
                color=RED
                if candle_data["Close"].item()>candle_data["Open"].item():
                    color=BLUE
                # for Price & Volume Class
                #print(volsec[j:j+1].iloc[0,0])
                #print(volsec[j:j+1])
                cv2.rectangle(img, (int(j*(SIZE/self.N)), int((-16+SIZE-40*candle_data["Close"].item())+0.5)), (int((j+1)*(SIZE/self.N)-1), int(-16+SIZE-40*candle_data["Open"].item()+0.5)), color, cv2.FILLED)
                cv2.line(img, (int(j*(SIZE/self.N))+int(SIZE/self.N/2)-1,int((-16+SIZE-40*candle_data["High"].item())+0.5)),(int(j*(SIZE/self.N))+int(SIZE/self.N/2)-1,int((-16+SIZE-40*candle_data["Low"].item())+0.5)),color)
                cv2.rectangle(img, (int(j*(SIZE/self.N)), int((SIZE-16*volsec[j:j+1].iloc[0,0])+0.5)-1), (int((j+1)*(SIZE/self.N)-1), SIZE), color, cv2.FILLED)
            imgs[:,:,:,i]=img
        return imgs

def main():
    df=pd.read_csv("finance_1d.csv")
    train_df=df[:5300]
    test_df=df[5300:]
    gen=PriceOnly(n=N)
    #gen=PriceAndVolume(n=N)
    #gen.generate(train_df)
    imgs=gen.generate(test_df)
    print(imgs.shape)
    for i in range(imgs.shape[3]):
        cv2.imwrite(f'priceOnly/input_{i}.png', imgs[:,:,:,i])
        #cv2.imwrite(f'priceAndVolume/input_{i}.png', imgs[:,:,:,i])
    #print(test_df[0:1]["Close"])


if __name__ == "__main__":
    main()
