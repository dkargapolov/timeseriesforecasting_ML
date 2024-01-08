from numpy import *
import matplotlib.pyplot as plt


def autocorelation(arr):
    lag_arr = arr[:-1]
    #yt - 1
    ytm1 = sum(lag_arr)

    #y1 middle
    y1_av = (sum(arr) - arr[0])/(len(arr)-1)
    #y2 middle
    y2_av = sum(lag_arr)/(len(arr)-1)
    #yt - y1 middle
    y3 = [x-y1_av for x in arr[1:]]

    #yt-1 - y2 middle
    y4 = [x-y2_av for x in lag_arr]
    #(yt - y1 middle)*(yt-1 - y2 middle)
    y5 = [x*y for x,y in zip(y3,y4)]
    y5s = sum(y5)
    #(yt - y1 middle)^2
    y6 = [x*x for x in y3]
    y6s = sum(y6)
    #(yt-1 - y2 middle)^2
    y7 = [x*x for x in y4]
    y7s = sum(y7)

    r = y5s/(sqrt(y6s*y7s))
    return r

def bGetting(y, t):

    
    y_av = sum(y)/len(y)
    t_av = sum(t)/len(t)
    #y*t
    yt = [a*b for a,b in zip(y,t)]
    yt_av = sum(yt)/len(yt)
    #t^2
    t2 = [x*x for x in t]
    t2_av = sum(t2)/len(t2)
    #b
    b = (yt_av - y_av * t_av)/(t2_av - t_av * t_av)
    return b

