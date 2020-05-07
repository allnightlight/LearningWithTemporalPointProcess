
import json
from datetime import datetime, timedelta
import matplotlib.pylab as plt
import numpy as np

myDatetimeFormat =  "%Y/%m/%d %H:%M"

def json_date_hook(json_dict):
    for key, val in json_dict.items():
        if isinstance(val, str):
            try:
                json_dict[key] = datetime.strptime(val, myDatetimeFormat)
            except:
                pass
    return json_dict

def json_converter(obj):
    if isinstance(obj, datetime):
        return datetime.strftime(obj, myDatetimeFormat)

def convertToJson(data):
    dataJson = json.dumps(data, default = json_converter, sort_keys = True)
    return dataJson

def parseFromJson(dataJson):
    data = json.loads(dataJson, object_hook = json_date_hook)
    return data

def drawColorMapOfEventProb(F, B, myXticklabel, myYticklabel, myTitle):

    Nx, Ny = F.shape
    assert F.shape == B.shape
    assert Nx == len(myXticklabel)
    assert Ny == len(myYticklabel)

    X = np.zeros(Nx*Ny)
    Y = np.zeros(Nx*Ny)
    S = np.zeros(Nx*Ny)

    cnt = 0
    for ix in range(Nx):
        for iy in range(Ny):
            X[cnt] = ix + 0.5
            Y[cnt] = iy + 0.5
            S[cnt] = F[ix, iy]
            cnt += 1

    fig = plt.gcf()
    ax = fig.add_subplot()

    ax.pcolor(B.T, cmap="Reds")
    ax.scatter(X, Y, s=S*120, marker="s", facecolor='red', edgecolor='white')

    ax.set_xticks(np.arange(0, Nx) + 0.5)
    ax.set_yticks(np.arange(0, Ny) + 0.5)
    ax.set_xticklabels(myXticklabel)
    ax.set_yticklabels(myYticklabel)
    #ax.grid()
    for k2 in range(Ny+1):
        ax.axhline(y=k2, c = "gray", linewidth=.5, linestyle= "--")
    for k1 in range(Nx+1):
        ax.axvline(x=k1, c = "gray", linewidth=.5, linestyle= "--")
    ax.set_xlim(0, Nx)
    ax.set_ylim(0, Ny)
    ax.set_title(myTitle)
    plt.tight_layout()
    
    return ax