import cv2
import numpy as np
import time

f1 = np.zeros([3,3]) #green (white on top)
f2 = np.zeros([3,3]) #red (white on top)
f3 = np.zeros([3,3]) #blue (white on top)
f4 = np.zeros([3,3]) #orange (white on top)
f5 = np.zeros([3,3]) #white (blue on top)
f6 = np.zeros([3,3]) #yellow (blue on top)
f3[:] = 3
f6[:] = 6
f1[:] = 1
f2[:] = 2
f4[:] = 4
f5[:] = 5

def R():
    global f1,f2,f3,f4,f5,f6
    temp = np.copy(f5)
    f5[:,2] = f1[:,2]
    f1[:,2] = np.fliplr(f6.T).T[:,0]
    f2 = np.rot90(f2,k=-1)
    f6[:,0] = f3[:,0]
    f3[:,0] = np.fliplr(temp.T).T[:,2]
    return 


def L():
    global f1,f2,f3,f4,f5,f6
    temp = np.copy(f5)
    f5[:,0] = np.fliplr(f3.T).T[:,2]
    f3[:,2] = f6[:,2]
    f4 = np.rot90(f4,k=-1)
    f6[:,2] = np.fliplr(f1.T).T[:,0]
    f1[:,0] = temp[:,0]
    return 


def F():
    global f1,f2,f3,f4,f5,f6
    temp = np.copy(f5)
    f1 = np.rot90(f1,k=-1)
    f5[2,:] = (np.fliplr(f4.T).T[:,2])
    f4[:,2] = (np.fliplr(f6)[2,:])
    f6[2,:] = (f2[:,0])
    f2[:,0] = (temp[2,:])
    return 


def B():
    global f1,f2,f3,f4,f5,f6
    temp = np.copy(f5)
    f3 = np.rot90(f3,k=-1)
    f5[0,:] = (f2[:,2])
    f2[:,2] = (f6[0,:])
    f6[0,:] = np.fliplr(f4.T).T[:,0]
    f4[:,0] = np.fliplr(temp)[0,:]
    return 


def U():
    global f1,f2,f3,f4,f5,f6
    temp = np.copy(f1)
    f5 = np.rot90(f5,k=-1)
    f1[0,:] = (f2[0,:])
    f2[0,:] = (f3[0,:])
    f3[0,:] = (f4[0,:])
    f4[0,:] = temp[0,:]
    return 


def D():
    global f1,f2,f3,f4,f5,f6
    temp = np.copy(f1)
    f6 = np.rot90(f6,k=-1)
    f1[2,:] = (f4[2,:])
    f4[2,:] = (f3[2,:])
    f3[2,:] = (f2[2,:])
    f2[2,:] = temp[2,:]
    return 


def R2():
    R()
    R()
    return

def L2():
    L()
    L()
    return

def F2():
    F()
    F()
    return
    
def B2():
    B()
    B()
    return
    
def U2():
    U()
    U()
    return
    
def D2():
    D()
    D()
    return

def R3():
    global f1,f2,f3,f4,f5,f6
    temp = np.copy(f5)
    f5[:,2] = np.fliplr(f3.T).T[:,0]
    f3[:,0] = f6[:,0]
    f6[:,0] = np.fliplr(f1.T).T[:,2]
    f1[:,2] = temp[:,2]
    f2 = np.rot90(f2,k=1)
    return 

def L3():
    global f1,f2,f3,f4,f5,f6
    temp = np.copy(f5)
    f5[:,0] = f1[:,0]
    f1[:,0] = np.fliplr(f6.T).T[:,2]
    f6[:,2] = f3[:,2]
    f3[:,2] = np.fliplr(temp.T).T[:,0]
    f4 = np.rot90(f4,k=1)
    return 

def F3():
    global f1,f2,f3,f4,f5,f6
    temp = np.copy(f5)
    f1 = np.rot90(f1,k=1)
    f5[2,:] = f2[:,0]
    f2[:,0] = f6[2,:]
    f6[2,:] = np.fliplr(f4.T).T[:,2]
    f4[:,2] = (np.fliplr(temp)[2,:])
    return 
    
def B3():
    global f1,f2,f3,f4,f5,f6
    temp = np.copy(f5)
    f3 = np.rot90(f3,k=1)
    f5[0,:] = np.fliplr(f4.T).T[:,0]
    f4[:,0] = np.fliplr(f6)[0,:]
    f6[0,:] = f2[:,2]
    f2[:,2] = temp[0,:]
    return 
    
def U3():
    global f1,f2,f3,f4,f5,f6
    temp = np.copy(f1)
    f5 = np.rot90(f5,k=1)
    f1[0,:] = (f4[0,:])
    f4[0,:] = (f3[0,:])
    f3[0,:] = (f2[0,:])
    f2[0,:] = temp[0,:]
    return 
    
def D3():
    global f1,f2,f3,f4,f5,f6
    temp = np.copy(f1)
    f6 = np.rot90(f6,k=1)
    f1[2,:] = (f2[2,:])
    f2[2,:] = (f3[2,:])
    f3[2,:] = (f4[2,:])
    f4[2,:] = temp[2,:]
    return 


move1 = [R,L,F,B,U,D]
move2 = [R2,L2,F2,B2,U2,D2]
move3 = [R3,L3,F3,B3,U3,D3]
movelist = [R,R2,R3,L,L2,L3,F,F2,F3,B,B2,B3,U,U2,U3,D,D2,D3]

def start(f1,f2,f3,f4,f5,f6):
    f3[:] = 3
    f6[:] = 6
    f1[:] = 1
    f2[:] = 2
    f4[:] = 4
    f5[:] = 5


def color(x,y,test,f):
    for j in range (0,9):
        a,b = int(j/3),int(j%3)
        if(f[a,b] == 1):
            test[x+30*a:x+30*a+29,y+30*b:y+30*b+29,0] = 0/255
            test[x+30*a:x+30*a+29,y+30*b:y+30*b+29,1] = 255/255
            test[x+30*a:x+30*a+29,y+30*b:y+30*b+29,2] = 0/255
        elif(f[a,b] == 2):
            test[x+30*a:x+30*a+29,y+30*b:y+30*b+29,0] = 0/255
            test[x+30*a:x+30*a+29,y+30*b:y+30*b+29,1] = 0/255
            test[x+30*a:x+30*a+29,y+30*b:y+30*b+29,2] = 255/255
        elif(f[a,b] == 3):
            test[x+30*a:x+30*a+29,y+30*b:y+30*b+29,0] = 255/255
            test[x+30*a:x+30*a+29,y+30*b:y+30*b+29,1] = 0/255
            test[x+30*a:x+30*a+29,y+30*b:y+30*b+29,2] = 0/255
        elif(f[a,b] == 4):
            test[x+30*a:x+30*a+29,y+30*b:y+30*b+29,0] = 0/255
            test[x+30*a:x+30*a+29,y+30*b:y+30*b+29,1] = 165/255
            test[x+30*a:x+30*a+29,y+30*b:y+30*b+29,2] = 255/255
        elif(f[a,b] == 5):
            test[x+30*a:x+30*a+29,y+30*b:y+30*b+29,0] = 255/255
            test[x+30*a:x+30*a+29,y+30*b:y+30*b+29,1] = 255/255
            test[x+30*a:x+30*a+29,y+30*b:y+30*b+29,2] = 255/255
        elif(f[a,b] == 6):
            test[x+30*a:x+30*a+29,y+30*b:y+30*b+29,0] = 0/255
            test[x+30*a:x+30*a+29,y+30*b:y+30*b+29,1] = 255/255
            test[x+30*a:x+30*a+29,y+30*b:y+30*b+29,2] = 255/255


def click(event,x,y,flags,param):
    global f1,f2,f3,f4,f5,f6,moves_made,moves,is3d
    if event == cv2.EVENT_LBUTTONDOWN:
        #print(x,y)
        if (y in range(240,271)):
            if(x in range(148,173)):
                F()
                moves_made=moves_made+'F '
                moves.append(6)
                if (is3d):
                    cube3(f1,f2,f3,f4,f5,f6)
                else:
                    cube(f1,f2,f3,f4,f5,f6)
            elif(x in range(183,208)):
                B()
                moves_made=moves_made+'B '
                moves.append(9)
                if (is3d):
                    cube3(f1,f2,f3,f4,f5,f6)
                else:
                    cube(f1,f2,f3,f4,f5,f6)
            elif(x in range(220,245)):
                R()
                moves_made= moves_made+'R '
                moves.append(15)
                if (is3d):
                    cube3(f1,f2,f3,f4,f5,f6)
                else:
                    cube(f1,f2,f3,f4,f5,f6)
            elif(x in range(257,282)):
                L()
                moves_made=moves_made+'L '
                moves.append(12)
                if (is3d):
                    cube3(f1,f2,f3,f4,f5,f6)
                else:
                    cube(f1,f2,f3,f4,f5,f6)
            elif(x in range(290,315)):
                U()
                moves_made=moves_made+'U '
                moves.append(0)
                if (is3d):
                    cube3(f1,f2,f3,f4,f5,f6)
                else:
                    cube(f1,f2,f3,f4,f5,f6)
            elif(x in range(328,353)):
                D()
                moves_made=moves_made+'D '
                moves.append(3)
                if (is3d):
                    cube3(f1,f2,f3,f4,f5,f6)
                else:
                    cube(f1,f2,f3,f4,f5,f6)
        elif (y in range(271,295)):
            if (x in range(145,160)):
                F2()
                moves_made=moves_made+'F2 '
                moves.append(7)
                if (is3d):
                    cube3(f1,f2,f3,f4,f5,f6)
                else:
                    cube(f1,f2,f3,f4,f5,f6)
            elif (x in range(161,176)):
                F3()
                moves_made=moves_made+"F' "
                moves.append(8)
                if (is3d):
                    cube3(f1,f2,f3,f4,f5,f6)
                else:
                    cube(f1,f2,f3,f4,f5,f6)
            elif (x in range(180,195)):
                B2()
                moves_made=moves_made+'B2 '
                moves.append(10)
                if (is3d):
                    cube3(f1,f2,f3,f4,f5,f6)
                else:
                    cube(f1,f2,f3,f4,f5,f6)
            elif (x in range(196,211)):
                B3()
                moves_made=moves_made+"B' "
                moves.append(11)
                if (is3d):
                    cube3(f1,f2,f3,f4,f5,f6)
                else:
                    cube(f1,f2,f3,f4,f5,f6)
            elif (x in range(217,232)):
                R2()
                moves_made=moves_made+'R2 '
                moves.append(16)
                if (is3d):
                    cube3(f1,f2,f3,f4,f5,f6)
                else:
                    cube(f1,f2,f3,f4,f5,f6)
            elif (x in range(233,248)):
                R3()
                moves_made=moves_made+"R' "
                moves.append(17)
                if (is3d):
                    cube3(f1,f2,f3,f4,f5,f6)
                else:
                    cube(f1,f2,f3,f4,f5,f6)
            elif (x in range(254,269)):
                L2()
                moves_made=moves_made+'L2 '
                moves.append(13)
                if (is3d):
                    cube3(f1,f2,f3,f4,f5,f6)
                else:
                    cube(f1,f2,f3,f4,f5,f6)
            elif (x in range(270,285)):
                L3()
                moves_made=moves_made+"L' "
                moves.append(14)
                if (is3d):
                    cube3(f1,f2,f3,f4,f5,f6)
                else:
                    cube(f1,f2,f3,f4,f5,f6)
            elif (x in range(287,302)):
                U2()
                moves_made=moves_made+'U2 '
                moves.append(1)
                if (is3d):
                    cube3(f1,f2,f3,f4,f5,f6)
                else:
                    cube(f1,f2,f3,f4,f5,f6)
            elif (x in range(303,318)):
                U3()
                moves_made=moves_made+"U' "
                moves.append(2)
                if (is3d):
                    cube3(f1,f2,f3,f4,f5,f6)
                else:
                    cube(f1,f2,f3,f4,f5,f6)
            elif (x in range(325,340)):
                D2()
                moves_made=moves_made+'D2 '
                moves.append(4)
                if (is3d):
                    cube3(f1,f2,f3,f4,f5,f6)
                else:
                    cube(f1,f2,f3,f4,f5,f6)
            elif (x in range(341,356)):
                D3()
                moves_made=moves_made+"D' "
                moves.append(5)
                if (is3d):
                    cube3(f1,f2,f3,f4,f5,f6)
                else:
                    cube(f1,f2,f3,f4,f5,f6)
        if y in range (20,56):
            if x in range(323,368):
                is3d = not(is3d)
                if (is3d):
                    cube3(f1,f2,f3,f4,f5,f6)
                else:
                    cube(f1,f2,f3,f4,f5,f6)
            
    return 


def cube(f1,f2,f3,f4,f5,f6):
    test = np.zeros((300,400,3))
    
    color(102,3,test,f1) 
    color(3,3,test,f5) 
    color(201,3,test,np.rot90(f6,k=2)) 
    color(102,102,test,f2)
    color(102,201,test,f3)
    color(102,300,test,f4)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.rectangle(test,(148,240),(172,270),(0.5,0.5,0.5),-1)
    cv2.rectangle(test,(183,240),(207,270),(0.5,0.5,0.5),-1)
    cv2.rectangle(test,(220,240),(244,270),(0.5,0.5,0.5),-1)
    cv2.rectangle(test,(257,240),(281,270),(0.5,0.5,0.5),-1)
    cv2.rectangle(test,(290,240),(314,270),(0.5,0.5,0.5),-1)
    cv2.rectangle(test,(328,240),(352,270),(0.5,0.5,0.5),-1)
    cv2.rectangle(test,(145,270),(159,294),(1,0.5,0.5),-1)
    cv2.rectangle(test,(161,270),(175,294),(1,0.5,0.5),-1)
    cv2.rectangle(test,(180,270),(194,294),(1,0.5,0.5),-1)
    cv2.rectangle(test,(196,270),(210,294),(1,0.5,0.5),-1)
    cv2.rectangle(test,(217,270),(231,294),(1,0.5,0.5),-1)
    cv2.rectangle(test,(233,270),(247,294),(1,0.5,0.5),-1)
    cv2.rectangle(test,(254,270),(268,294),(1,0.5,0.5),-1)
    cv2.rectangle(test,(270,270),(284,294),(1,0.5,0.5),-1)
    cv2.rectangle(test,(287,270),(301,294),(1,0.5,0.5),-1)
    cv2.rectangle(test,(303,270),(317,294),(1,0.5,0.5),-1)
    cv2.rectangle(test,(325,270),(339,294),(1,0.5,0.5),-1)
    cv2.rectangle(test,(341,270),(355,294),(1,0.5,0.5),-1)
    cv2.putText(test,'F B R L U D',(150,265), font, 1,(255/255,255/255,255/255),3,cv2.LINE_AA)
    cv2.putText(test,"2 ' 2 ' 2 ' 2 ' 2 ' 2 '",(145,290), font, 0.6,(255/255,255/255,255/255),1,cv2.LINE_AA)
    cv2.rectangle(test,(323,20),(368,55),(1,0.5,0.5),-1)
    cv2.putText(test,'3D',(325,50), font, 1,(255/255,255/255,255/255),2,cv2.LINE_AA)
    
    cv2.namedWindow("cube")
    cv2.setMouseCallback("cube",click)
    cv2.imshow("cube",test)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def color3(x):
    test=np.zeros(3)
    if(x == 1):
        test[0] = 0/255
        test[1] = 255/255
        test[2] = 0/255
    elif(x == 2):
        test[0] = 0/255
        test[1] = 0/255
        test[2] = 255/255
    elif(x == 3):
        test[0] = 255/255
        test[1] = 0/255
        test[2] = 0/255
    elif(x == 4):
        test[0] = 0/255
        test[1] = 165/255
        test[2] = 255/255
    elif(x == 5):
        test[0] = 255/255
        test[1] = 255/255
        test[2] = 255/255
    elif(x == 6):
        test[0] = 0/255
        test[1] = 255/255
        test[2] = 255/255
    return test

def face1(test,f):
    for i in range(3):
        for j in range(3):
            col = color3(f[i,j])
            cv2.rectangle(test,(30+40*j,100+40*i),(30+40*j+38,100+40*i+38),col,-1)
            
def face2(test,f):
    for i in range(3):
        for j in range(3):
            col = color3(f[i,j])
            contours = np.array( [ [150+20*j,99+40*i-20*j], [150+20*j,99+40*i+38-20*j], [150+20*j+18,99+40*i+20-20*j], [150+20*j+18,99+40*i-18-20*j] ] )
            cv2.fillPoly(test, pts =[contours], color=col)    
            
def face3(test,f):
    for i in range(3):
        for j in range(3):
            col = color3(f[i,j])
            contours = np.array( [ [89+40*j-20*i,40+20*i], [89+40*j-20*i+38,40+20*i], [89+40*j-20*i+20,40+20*i+18], [89+40*j-18-20*i,40+20*i+18] ] )
            cv2.fillPoly(test, pts =[contours], color=col) 


def cube3(f1,f2,f3,f4,f5,f6):
    test = np.zeros((300,400,3))
    face1(test,f1)
    face2(test,f2)
    face3(test,f5)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.rectangle(test,(148,240),(172,270),(0.5,0.5,0.5),-1)
    cv2.rectangle(test,(183,240),(207,270),(0.5,0.5,0.5),-1)
    cv2.rectangle(test,(220,240),(244,270),(0.5,0.5,0.5),-1)
    cv2.rectangle(test,(257,240),(281,270),(0.5,0.5,0.5),-1)
    cv2.rectangle(test,(290,240),(314,270),(0.5,0.5,0.5),-1)
    cv2.rectangle(test,(328,240),(352,270),(0.5,0.5,0.5),-1)
    cv2.rectangle(test,(145,270),(159,294),(1,0.5,0.5),-1)
    cv2.rectangle(test,(161,270),(175,294),(1,0.5,0.5),-1)
    cv2.rectangle(test,(180,270),(194,294),(1,0.5,0.5),-1)
    cv2.rectangle(test,(196,270),(210,294),(1,0.5,0.5),-1)
    cv2.rectangle(test,(217,270),(231,294),(1,0.5,0.5),-1)
    cv2.rectangle(test,(233,270),(247,294),(1,0.5,0.5),-1)
    cv2.rectangle(test,(254,270),(268,294),(1,0.5,0.5),-1)
    cv2.rectangle(test,(270,270),(284,294),(1,0.5,0.5),-1)
    cv2.rectangle(test,(287,270),(301,294),(1,0.5,0.5),-1)
    cv2.rectangle(test,(303,270),(317,294),(1,0.5,0.5),-1)
    cv2.rectangle(test,(325,270),(339,294),(1,0.5,0.5),-1)
    cv2.rectangle(test,(341,270),(355,294),(1,0.5,0.5),-1)
    cv2.putText(test,'F B R L U D',(150,265), font, 1,(255/255,255/255,255/255),3,cv2.LINE_AA)
    cv2.putText(test,"2 ' 2 ' 2 ' 2 ' 2 ' 2 '",(145,290), font, 0.6,(255/255,255/255,255/255),1,cv2.LINE_AA)
    cv2.rectangle(test,(323,20),(368,55),(1,0.5,0.5),-1)
    cv2.putText(test,'2D',(325,50), font, 1,(255/255,255/255,255/255),2,cv2.LINE_AA)
    
    cv2.namedWindow("cube")
    cv2.setMouseCallback("cube",click)
    cv2.imshow("cube",test)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

start (f1,f2,f3,f4,f5,f6)
moves_made = ' '
moves = []
is3d = True
cube3(f1,f2,f3,f4,f5,f6)
print ("moves made :",moves_made)


tstart = time.time()

facenames = ["U", "D", "F", "B", "L", "R"]
affected_cubies = [[0, 1, 2, 3, 0, 1, 2, 3], [4, 7, 6, 5, 4, 5, 6, 7], [0, 9, 4, 8, 0, 3, 5, 4], [2, 10, 6, 11, 2, 1, 7, 6], [3, 11, 7, 9, 3, 2, 6, 5], [1, 8, 5, 10, 1, 0, 4, 7]]
phase_moves = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17], [0, 1, 2, 3, 4, 5, 7, 10, 12, 13, 14, 15, 16, 17], [0, 1, 2, 3, 4, 5, 7, 10, 13, 16], [1, 4, 7, 10, 13, 16]]

def move_str(move):
    return facenames[int(move/3)]+{1: '', 2: '2', 3: "'"}[move%3+1]

class cube_state:
    def __init__(self, state, route=None):
        self.state = state
        self.route = route or []

    def id_(self, phase):
        if phase == 0:
            return tuple(self.state[20:32])
        elif phase == 1:
            result = self.state[31:40]
            for e in range(12):
                result[0] |= int((self.state[e] / 8)) << e;
            return tuple(result)
        elif phase == 2:
            result = [0,0,0]
            for e in range(12):
                result[0] |= (2 if (self.state[e] > 7) else (self.state[e] & 1)) << (2*e)
            for c in range(8):
                result[1] |= ((self.state[c+12]-12) & 5) << (3*c)
            for i in range(12, 20):
                for j in range(i+1, 20):
                    result[2] ^= int(self.state[i] > self.state[j])
            return tuple(result)
        else:
            return tuple(self.state)

    def apply_move(self, move):
        face, turns = int(move / 3), move % 3 + 1
        newstate = self.state[:]
        for turn in range(turns):
            oldstate = newstate[:]
            for i in range(8):
                isCorner = int(i > 3)
                target = affected_cubies[face][i] + isCorner*12
                killer = affected_cubies[face][(i-3) if (i&3)==3 else i+1] + isCorner*12
                orientationDelta = int(1<face<4) if i<4 else (0 if face<2 else 2 - (i&1))
                newstate[target] = oldstate[killer]
                newstate[target+20] = oldstate[killer+20] + orientationDelta
                if turn == turns-1:
                    newstate[target+20] %= 2 + isCorner
        return cube_state(newstate, self.route+[move])

goal_state = cube_state(list(range(20))+20*[0])
state = cube_state(goal_state.state[:])


for move in moves:
    state = state.apply_move(move)
state.route = []

print ('solving')

for phase in range(4):
    current_id, goal_id = state.id_(phase), goal_state.id_(phase)
    states = [state]
    state_ids = set([current_id])
    if current_id != goal_id:
        phase_ok = False
        while not phase_ok:
            next_states = []
            for cur_state in states:
                for move in phase_moves[phase]:
                    next_state = cur_state.apply_move(move)
                    next_id = next_state.id_(phase)
                    if next_id == goal_id:
                        print(','.join([move_str(m) for m in next_state.route]) + ' (%d moves)'% len(next_state.route) + ' Phase:%d' %(phase+1))
                        phase_ok = True
                        state = next_state
                        break
                    if next_id not in state_ids:
                        state_ids.add(next_id)
                        next_states.append(next_state)
                if phase_ok:
                    break
            states = next_states

print(time.time()-tstart)


