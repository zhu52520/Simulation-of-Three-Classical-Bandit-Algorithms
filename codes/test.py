import random
import math
from scipy.stats import beta
from tqdm.notebook import tqdm


class Result():
    def __init__(self, repeat):
        self.repeat = repeat #repeat times
        self.count = 0 #experiment times
        self.theta_total = [0,0,0]
        self.gain_total = 0

    def greedy(self, epsilon, N):
        theta_mean = [0,0,0] # estimation of theta_j
        count = [0,0,0] # count(j)
        gain = 0 # total reward
        result = 0 # the result of the arm of this time
        for t in range(N):
            # 生成I(t)
            if random.random() <= epsilon:
                I = random.randint(1,3)
            else:
                I = arg_max(theta_mean) + 1
            result = arms(I)
            gain += result
            count[I-1] += 1
            theta_mean[I-1] += (result-theta_mean[I-1])/count[I-1]
        self.count += 1
        for i in range(3):
            self.theta_total[i] += theta_mean[i]
        self.gain_total += gain

    def UCB(self, c, N):
        theta_mean = [0,0,0] # estimation of theta_j
        count = [0,0,0] # count(j)
        gain = 0 # total reward
        result = 0 # the result of the arm of this time
        for t in range(3):
            count[t] += 1
            theta_mean[t] = arms(t+1)
        for t in range(3, N):
            temp = [0,0,0]
            for j in range(3):
                temp[j] = theta_mean[j] + c * (2*math.log10(t)/count[j])**0.5
            I = arg_max(temp)
            result = arms(I)
            gain += result
            count[I-1] += 1
            theta_mean[I-1] += (result-theta_mean[I-1])/count[I-1]
        self.count += 1
        for i in range(3):
            self.theta_total[i] += theta_mean[i]
        self.gain_total += gain

    def TS(self, index, N):
        if index == 1:
            parameter = [[1,1],[1,1],[1,1]]
        elif index == 2:
            parameter = [[601,401],[401,601],[2,3]]
        else:
            raise IndexError
        theta_mean = [0,0,0] # estimation of theta_j
        gain = 0 # total reward
        result = 0 # the result of the arm of this time
        for t in range(N):
            sample_theta = [0,0,0]
            for i in range(3):
                sample_theta[i] = beta.rvs(parameter[i][0],parameter[i][1])
            I = arg_max(sample_theta)
            print(I)
            result = arms(I)
            parameter[I-1][0] += result
            parameter[I-1][1] += 1-result
            gain += result
        for i in range(3):
            theta_mean[i] = parameter[i][0] / (parameter[i][0]+parameter[i][1])
        self.count += 1
        for i in range(3):
            self.theta_total[i] += theta_mean[i]
        self.gain_total += gain
        

    
    def output(self):
        gain_mean = self.gain_total / self.count
        print("the average gain of {} times experiment is {}".format(self.count, gain_mean))

        theta_mean = [0,0,0]
        for i in range(3):
            theta_mean[i] = self.theta_total[i] / self.count
        print("the estimated probability is {}".format(theta_mean))
    


def arms(index, p1 = 0.8, p2 = 0.6, p3 = 0.5):
    '''
    返回第index个摇臂拉下的结果
    '''
    if index == 1:
        if random.random() <= p1:
            return 1
        else:
            return 0
    elif index ==2:
        if random.random() <= p2:
            return 1
        else:
            return 0
    elif index == 3:
        if random.random() <= p3:
            return 1
        else:
            return 0
    else:
        raise IndexError

def arg_max(list):
    '''
    返回列表最大值的索引
    当列表存在多个最大值时，等概率返回其中一个
    '''
    index = 0
    max_value = 0
    max_index = []
    for i in list:
        if i > max_value:
            max = i
            max_index = [index]
        elif i == max_value:
            max_index.append(index)
        index += 1
    if len(max_index) == 1:
        return max_index[0]
    else:
        return random.choice(max_index)


T = Result(200)
for i in tqdm(range(20)):
    T.TS(1,20)
T.output()

