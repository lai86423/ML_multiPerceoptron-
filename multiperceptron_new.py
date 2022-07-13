import tkinter.ttk as ttk
import tkinter as tk
import tkinter.font as tkfont
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

def Multiperceoptron():
    #接收使用者輸入
    round = int(ent_round.get())
    rate = float(ent_ratio.get()) / (1 + (round /30))
    neu = int(ent_neu.get())
    file = str(file_name.get())
    #資料處理 分成陣列、隨機打亂、訓練與測試資料分割
    np_file = np.genfromtxt(file, delimiter=' ')
    np.random.shuffle(np_file)
    shape = list(np_file.shape)
    num_train = int(shape[0] * 2 / 3)
    #期望輸出設定、加入bias
    file_detail,expect_out = data_processing(np_file,shape)
    #隨機初始鍵結值
    weight = init_weight(neu,shape)
    #訓練開始 [forward/backward /計算delta /調整鍵結值 /計算辨識率]
    best_accu,best_wei,best_los = training(weight,round,file_detail,shape,expect_out,num_train,rate)
    #測試資料訓練 計算辨識率
    test_acc = test_accu(best_wei,file_detail,num_train,expect_out,shape)
    #畫圖
    plot_figure(best_wei,neu,np_file,num_train)
    
    #介面變數設定
    accuracy_label1.config(text = '訓練辨識率')
    accuracy_label2.config(text = best_accu)
    accuracy_label3.config(text = '')
    accuracy_label4.config(text = '')
    accuracy_label3.config(text = '測試辨識率')
    accuracy_label4.config(text = test_acc)
    RMSE_label1.config(text = 'RMSE')
    RMSE_label2.config(text = best_los)
    weight_label1.config(text = '鍵結值')
    weight_label2.delete('1.0','end')
    weight_label2.insert('insert',best_wei)
    
    plt.show()

def data_processing(np_file,shape):
    
    expect = np_file[:,shape[1] - 1].reshape((shape[0],1))
    #去除期望輸出
    file_detail = np.delete(np_file,shape[1]-1,1)
    #插入bias
    a = np.linspace(-1,-1,shape[0])
    file_detail = np.insert(file_detail,0,values=a,axis=1)
    #期望輸出2改0
    for i in range(shape[0]):
        if expect[i] == 2:
            expect[i] = 0

    return file_detail,expect

def init_weight(neu,shape):
    #使用dictionary建立鍵結值資料
    weight = {}
    #依據神經元數目和資料維度建立隱藏層weight
    weight[0] = np.random.randn(shape[1],neu) 
    #輸出層weight維度為神經元數目+1 數量為1
    weight[1] = np.random.randn(neu+1,1)
    return weight
    
def training(weight,round,file_detail,shape,expect_out,num_train,rate):
    right = []
    loss = []
    final_round = 0
    best_accu = 0
    for i in range(round):
        for j in range(num_train):
            #前饋計算輸出
            threshold = forward(j,weight,file_detail)
            #倒傳遞計算delta
            delta = backward(j,weight,file_detail,expect_out,threshold)
            #依delta更正weight
            weight = modi_weight(j,weight,file_detail,threshold,delta,rate)
        print('迴圈',i + 1)
        #計算辨識率
        accuracy,right2,loss2,error = cal_accuracy(weight,file_detail,num_train,expect_out,right,loss)
        
        final_round = i
        #達最佳辨識率跳離迴圈
        if accuracy >= best_accu:
            best_accu = accuracy
            best_wei = weight.copy()
            best_los = error
        if accuracy == 100:
            break
    #辨識率與誤差線圖範圍等設定
    plt.figure(figsize=(6,6))
    plt1 = plt.subplot(221)
    xlim = [i for i in range(1,final_round+2)]
    plt1.set_title('accuracy')
    plt1.plot(xlim,right2,'b')
    plt2 = plt.subplot(222)
    plt2.set_title('loss function')
    plt2.plot(xlim,loss2,'r')
    return best_accu,best_wei,best_los

def forward(j,weight,file_detail):
    threshold = {}
    threshold[0] = sigmoid(file_detail[j].dot(weight[0]))
    threshold[0] = np.insert(threshold[0],0,[-1])
    threshold[1] = threshold[0].dot(weight[1])
    threshold[1] = sigmoid(threshold[1])
    return threshold        

def sigmoid(threshold):
    for idx,layer in enumerate(threshold):
        threshold[idx] = 1 / (1 + np.exp(-layer))   
    return threshold

def backward(j,weight,file_detail,expect_out,threshold):
    delta = {}
    delta[1] = (expect_out[j] - threshold[1]) * threshold[1] * (np.ones(1) - threshold[1])
    delta[0] = threshold[0][1::] * (np.ones(np.size(threshold[0]) - 1) - threshold[0][1::]) * sigma(weight,threshold,0,delta)  
    return delta

def sigma(weight,threshold,k,delta):
    total = np.ones(np.size(threshold[k]) - 1)
    for idx in range(np.size(threshold[k])-1):
        total[idx] = np.sum(delta[k+1] * weight[k+1][idx+1])
        
    return total
def modi_weight(j,weight,file_detail,threshold,delta,rate):
    
    weight[0] = weight[0].transpose() + rate * delta_weight(delta[0],file_detail[j])
    weight[0] = weight[0].transpose()
    
    weight[1] = weight[1].transpose() + rate * delta_weight(delta[1],threshold[0])
    weight[1] = weight[1].transpose()
    return weight

def delta_weight(delta,input):
    new = np.zeros((np.size(delta),np.size(input)))
    
    for i in range(np.size(delta)):
        new[i] = delta[i] * input
    
    return new

def cal_accuracy(weight,file_detail,num_train,expect_out,right,loss):
    plt.close('all')
    b = 0
    error = 0
    for i in range(num_train):
        
        a = sigmoid(file_detail[i].dot(weight[0]))
        a = np.insert(a,0,[-1])
        a = a.dot(weight[1])
        a = sigmoid(a)

        #期望輸出跟實際輸出比對 算誤差
        error += (np.sum((expect_out[i] - a) * (expect_out[i] - a))) 
        out = np.around(a)
        if (out == expect_out[i]).all():
            b += 1
    error = (error / 2) / num_train        
    accuracy =  (b / num_train) * 100 
    
    right.append(accuracy)
    loss.append(error)
    
    print('準確率',accuracy,'%  誤差',error)
    return accuracy,right,loss,error

def plot_figure(weight,neu,np_file,num_train):
    x = np_file[:,0]
    x_max = int(x.max())+1
    x_min = int(x.min())-1
    y = np_file[:,1]
    y_max = int(y.max())+1
    y_min = int(y.min())-1
    z = np_file[:,2]
    plt3 = plt.subplot(223)
    plt3.set_title('bivariate distribution')
    plt3.set_xlim(x_min,x_max)
    plt3.set_ylim(y_min,y_max)
    for x1 in np.arange(x_min,x_max,(x_max - x_min) / 60):
        for x2 in np.arange(y_min,y_max,(y_max - y_min) / 60):
            
            point = np.array([-1,x1,x2])
            a = sigmoid(point.dot(weight[0]))
            a = np.insert(a,0,[-1])
            
            a = a.dot(weight[1])
            a = sigmoid(a)
            a = np.around(a)
            
            if (a == [1]).all():
                plt3.plot(x1,x2,color='cyan',linestyle='',marker='s',ms=5)
            elif(a == [0]).all():
                plt3.plot(x1,x2,color='navajowhite',linestyle='',marker='s',ms=5)
            

    for i in range(num_train):
        if np_file[i][2] == 1:
            plt3.plot(x[i],y[i],color='dodgerblue',linestyle='',marker='o',ms=1)
        elif np_file[i][2] == 2 or np_file[i][2] == 0:
            plt3.plot(x[i],y[i],color='forestgreen',linestyle='',marker='o',ms=1)

def test_accu(weight,file_detail,num_train,expect_out,shape):
    b = 0
    for i in range(num_train,shape[0]):
        
        a = sigmoid(file_detail[i].dot(weight[0]))
        a = np.insert(a,0,[-1])
        a = a.dot(weight[1])
        a = sigmoid(a)
        out = np.around(a)
        
        if (out == expect_out[i]).all():
            b += 1        
    accuracy =  (b / (shape[0]-num_train)) * 100 
    return accuracy

#GUI
#介面基本設定
window = tk.Tk()
window.geometry('500x600')
window.title('HW2 Multiperceptron')

# 字體
default_front = tkfont.nametofont("TkDefaultFont")
default_front.configure(size=10)

#顯示選擇檔案
label_top = tk.Label(window,text = "Choose training file")
label_top.pack()

#檔案選擇設定
file_option = ('perceptron1.txt','perceptron1.txt','perceptron2.txt','2Ccircle1.txt','2Circle1.txt','2Circle2.txt',
             '2CloseS.txt','2CloseS2.txt','2CloseS3.txt','2cring.txt','2CS.txt','2Hcircle1.txt','2ring.txt')
file_name = tk.StringVar()

#下拉選單
##callback Funtion
def callbackFunc_showselect(event):
    selected_file = event.widget.get()
    lab_result.config(text=selected_file) 

##下拉選單設定
combox = ttk.Combobox(window,values=file_option,textvariable=file_name, font=default_front)
combox.bind('<<ComboboxSelected>>',callbackFunc_showselect)
combox.current(0)
combox.pack()
##顯示已選檔案
lab_result = tk.Label(window, font=default_front, fg='royal blue', width=18)
lab_result.pack()


#建立ratio_frame群組
ratio_frame = tk.Frame(window)
ratio_frame.pack(side=tk.TOP)
#學習率
ratio_label = tk.Label(ratio_frame,text='輸入學習率(小數)')
ratio_label.pack(side=tk.LEFT)
#輸入學習率
ent_ratio = tk.Entry(ratio_frame)
ent_ratio.pack(side=tk.LEFT)

#建立round_frame群組
round_frame = tk.Frame(window)
round_frame.pack(side=tk.TOP)
# 收斂條件
round_label = tk.Label(round_frame,text='設定收斂條件 - 訓練次數')
round_label.pack(side=tk.LEFT)
#輸入收斂次數
ent_round = tk.Entry(round_frame)
ent_round.pack(side=tk.LEFT)

#建立neu_frame群組
neu_frame = tk.Frame(window)
neu_frame.pack(side=tk.TOP)
#神經元數設定
neu_label = tk.Label(neu_frame,text='神經元')
neu_label.pack(side=tk.LEFT)
#輸入神經元數
ent_neu = tk.Entry(neu_frame)
ent_neu.pack(side=tk.LEFT)

#確定開始鍵
cal_button = tk.Button(window,text='開始',command=Multiperceoptron)
cal_button.pack()

#顯示訓練與測試辨識率
acc_frame = tk.Frame(window)
acc_frame.pack(side=tk.TOP)
accuracy_label1 = tk.Label(acc_frame)
accuracy_label1.pack(side=tk.LEFT)
accuracy_label2 = tk.Label(acc_frame)
accuracy_label2.pack(side=tk.LEFT)
accuracy_label3 = tk.Label(acc_frame)
accuracy_label3.pack(side=tk.LEFT)
accuracy_label4 = tk.Label(acc_frame)
accuracy_label4.pack(side=tk.LEFT)

#顯示訓練與測試均方根誤差
RMSE_frame = tk.Frame(window)
RMSE_frame.pack(side=tk.TOP)
RMSE_label1 = tk.Label(RMSE_frame)
RMSE_label1.pack(side=tk.LEFT)
RMSE_label2 = tk.Label(RMSE_frame)
RMSE_label2.pack(side=tk.LEFT)

#顯示鍵結值
scroll = tk.Scrollbar()
weight_frame = tk.Frame(window)
weight_frame.pack(side=tk.TOP)
weight_label1 = tk.Label(weight_frame)
weight_label1.pack(side=tk.LEFT)

weight_label2 = tk.Text(weight_frame,width='50',height='20')
weight_label2.pack(side=tk.LEFT)
scroll.config(command=weight_label2.yview)
weight_label2.config(yscrollcommand=scroll.set)

window.mainloop()