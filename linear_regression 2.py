import numpy as np
import matplotlib.pyplot as plt
import csv 

points = np.array([])
with open('data.csv') as f:
    data = csv.reader(f)
    points = next(data)
    for row in data:
        points = np.vstack((points,row))
        

# points = np.random.rand(100, 2)
# y = w*x + b 
def loss(points,w=0,b=0): # y:actual points:(x,y)
    total_loss = 0
    for i in range(len(points)):
        x = float(points[i,0])
        y = float(points[i,1])
        total_loss += (( w * x + b ) - y) ** 2
    total_loss/=float(len(points))
    return total_loss
# print(loss(points))

# learning rate 不可以太大，容易發散
def upload(w,b,points,learning_rate=0.001):
    dloss_dw = 0
    dloss_db = 0
    N = float(len(points))
    for i in range(len(points)):
        x = float(points[i,0])
        y = float(points[i,1])
        dloss_dw += (2 / N) * x * ((w * x + b) - y)
        dloss_db += (2 / N) * ((w * x + b) - y)
    w_prime = w - learning_rate * dloss_dw
    b_prime = b - learning_rate * dloss_db
    return [w_prime,b_prime]

def gradient_decent_runner(points,iteration=1000,learning_rate=0.00001,starting_w=0,starting_b=0):
    w = starting_w
    b = starting_b
    for i in range(iteration):

        w , b = upload(w,b,points,learning_rate)
        print("loss: ",loss(points,w,b))
    return [w,b]

# w , b = gradient_decent_runner(points)
def find_max_min(points,w,b):
    max_x, min_x = float(points[0,0]) ,float(points[0,0])
    for i in range(len(points)):
        if float(points[i,0]) > max_x:
            max_x = float(points[i,0]) 
        if float(points[i,0]) < min_x:
            min_x = float(points[i,0])
    max_y = w * max_x + b
    min_y = w * min_x + b
    return [min_x,min_y,max_x,max_y]



def gradient_animation(points,iteration=200,learning_rate=0.00001,starting_w=0,starting_b=0):
    w = starting_w
    b = starting_b
    fig = plt.figure(figsize=(12,8))
    for i in range(iteration):
        plt.cla()
        plt.xlim(0,80)
        plt.ylim(15,120)
        w , b = upload(w,b,points,learning_rate)
        print("loss: ",loss(points,w,b))
        for i in range(len(points)):
            plt.plot(float(points[i,0]),float(points[i,1]),'o',color='red')
        min_x, min_y, max_x, max_y = find_max_min(points,w,b)
        plt.plot([min_x, max_x],[min_y, max_y],color='green')
        plt.pause(0.0001)
        plt.ion()

gradient_animation(points)
