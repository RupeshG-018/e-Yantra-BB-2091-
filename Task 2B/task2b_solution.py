def sysCall_init():
    sim = require('sim')

    # do some initialization here
    # This function will be executed once when the simulation starts
    
    # Instead of using globals, you can do e.g.:
    # self.myVariable = 21000000
    
    ######## ADD YOUR CODE HERE #######
    # Hint: Initialize the scene objects which you will require
    #       Initialize algorithm related variables heres
    self.body = sim.getObject('/body')
    self.cone=sim.getObject('/Cone')
    self.floor = sim.getObject('/Floor')
    self.right_joint = sim.getObject('/right_joint')
    self.left_joint = sim.getObject('/left_joint')
    self.right_wheel = sim.getObject('/right_wheel')
    self.left_wheel = sim.getObject('/left_wheel')
    
    self.prismatic_joint = sim.getObject('/Prismatic_joint')
    self.arm_joint = sim.getObject('/arm_joint')

    #State Variables
    self.x = 0
    self.x_dot = 0
    self.theta_right_wheel = 0
    self.theta_left_wheel = 0
    self.theta = 0
    self.theta_dot = 0
    self.phi = 0
    self.phi_dot = 0
    self.x1 = 0
    self.x2 = 0
    self.x3 = 0
    self.x4 = 0
    self.x5 = 0
    self.x6 = 0
    self.U = 0
    self.Ur = 0
    self.Ul = 0 
    self.key1 = ''
    self.key2 = ''
    #self.step =1
    self.arm_rotation = 5
    self.step_down = 1
    self.step_up =1
    self.down_count = 0
    self.up_count =0
    self.out_count = 0
    self.step_out = 1
    self.in_count = 0
    self.step_in = 1
    self.prismatic_movement = 5
    self.phi1 = 0
    #self.pairs = []
    
    ##################################
    pass
    
def sysCall_actuation():
    # put your actuation code here
    # This function will be executed at each simulation time step

    ####### ADD YOUR CODE HERE ######
    # Hint: Use the error feedback and apply control algorithm here
    #       Provide the resulting actuation as input to the actuator joint
    # Example psuedo code:
    #   x1 = error_state_1; # Error in states w.r.t desired setpoint
    #   x2 = error_state_2;
    #   x3 = error_state_3;
    #   x4 = error_state_4;
    #   k = [gain_1 , gain_1, gain_3, gain_4];      # These gains will be generated by control algorithm. For ex: LQR, PID, etc.
    #   U = -k[1]*x1 +k[2]*x2 -k[3]*x3 +k[4]*x4;    # +/- Sign convention may differ according to implementation
    #   Set_joint_actuation(U);                     # Provide this calculated input to system's actuator

    self.x1 = -self.theta
    self.x2 = -self.theta_dot
    self.x3 = -self.phi
    self.x4 = -self.phi_dot
    
    k = [-0.44721,-1.5028,-64.755,-6.4064]
    self.U = k[0]*self.x1 + k[1]*self.x2 + k[2]*self.x3 + k[3]*self.x4
    
    sim.setJointTargetVelocity(self.right_joint , -self.U + self.Ur)
    sim.setJointTargetVelocity(self.left_joint , -self.U + self.Ul)
    step =1
    
    if self.key1 == 'forward': #Forward
        self.Ur-=10
        self.Ul-=10
    elif self.key1 == 'backward': #Backward
        self.Ur+=10
        self.Ul+=10
    elif self.key1 == 'left': #Left
        self.Ur-=2.5
        self.Ul+=2.5
    elif self.key1 == 'right': #Right
        self.Ur+=2.5
        self.Ul-=2.5
    self.key1 = ''
    
    if self.key2 == 'gclose': #Gripper Close
        if(self.step_in < self.prismatic_movement*self.in_count):
            sim.setJointTargetVelocity(self.prismatic_joint, 0.07)
            self.step_in +=1
        elif(self.step_in == self.prismatic_movement*self.in_count):
            sim.setJointTargetVelocity(self.prismatic_joint,0)
    elif self.key2 == 'gopen': #Gripper Open
        if(self.step_out < self.prismatic_movement*self.out_count):
            sim.setJointTargetVelocity(self.prismatic_joint, -0.07)
            self.step_out +=1
        elif(self.step_out == self.prismatic_movement*self.out_count):
            sim.setJointTargetVelocity(self.prismatic_joint,0)
    elif self.key2 == 'gdown': #Gripper Down
        if(self.step_down < self.arm_rotation*self.down_count):
            sim.setJointTargetVelocity(self.arm_joint,1)
            self.step_down +=1
        elif(self.step_down == self.arm_rotation*self.down_count):
            sim.setJointTargetVelocity(self.arm_joint,0)
    elif self.key2 == 'gup': #Gripper Up
        if(self.step_up <self.arm_rotation*self.up_count):
            sim.setJointTargetVelocity(self.arm_joint,-1)
            self.step_up +=1
        elif(self.step_up == self.arm_rotation*self.up_count):
            sim.setJointTargetVelocity(self.arm_joint,0)
            
    print(self.step_out)

    #################################
    pass


def sysCall_sensing():
    # put your sensing code here
    # This function will be executed at each simulation time step
    
    ####### ADD YOUR CODE HERE ######
    # Hint: Take feedback here & do the error calculation
    self.x = sim.getObjectPosition(self.body, sim.handle_world)[1]
    self.x_dot = sim.getObjectVelocity(self.body)[0][1]
    self.theta_right_wheel=sim.getObjectOrientation(self.right_wheel, self.floor)[2]
    self.theta_left_wheel=sim.getObjectOrientation(self.left_wheel, self.floor)[2]
    self.theta = (self.theta_right_wheel + self.theta_left_wheel)/2
    self.phi1 = sim.getObjectOrientation(self.body,self.floor)[2]

    if  self.phi1 < 0.6 and self.phi1 > -0.6:
        self.theta_dot_right_wheel = sim.getVelocity(self.right_wheel)[1][0]
        self.theta_dot_left_wheel = sim.getVelocity(self.left_wheel)[1][0]
        self.theta_dot = (self.theta_dot_right_wheel + self.theta_dot_left_wheel)/2
        self.phi = sim.getObjectOrientation(self.body,self.floor)[0]
        self.phi_dot = sim.getVelocity(self.body)[1][0]
    
    if self.phi1 > 0.6 and self.phi1 <= 2:
        self.theta_dot_right_wheel = sim.getVelocity(self.right_wheel)[1][1]
        self.theta_dot_left_wheel = sim.getVelocity(self.left_wheel)[1][1]
        self.theta_dot = (self.theta_dot_right_wheel + self.theta_dot_left_wheel)/2
        self.phi = sim.getObjectOrientation(self.body,self.floor)[1]
        self.phi_dot = sim.getVelocity(self.body)[1][1]
    
    if self.phi1 > 2 or self.phi1 < -2:
        self.theta_dot_right_wheel = -sim.getVelocity(self.right_wheel)[1][0]
        self.theta_dot_left_wheel = -sim.getVelocity(self.left_wheel)[1][0]
        self.theta_dot = (self.theta_dot_right_wheel + self.theta_dot_left_wheel)/2
        self.phi = -sim.getObjectOrientation(self.body,self.floor)[0]
        self.phi_dot = -sim.getVelocity(self.body)[1][0]
        
    if self.phi1 < -0.6 and self.phi1 >= -2:
        self.theta_dot_right_wheel = -sim.getVelocity(self.right_wheel)[1][1]
        self.theta_dot_left_wheel = -sim.getVelocity(self.left_wheel)[1][1]
        self.theta_dot = (self.theta_dot_right_wheel + self.theta_dot_left_wheel)/2
        self.phi = -sim.getObjectOrientation(self.body,self.floor)[1]
        self.phi_dot = -sim.getVelocity(self.body)[1][1]
        
    
    message,data,data2 = sim.getSimulatorMessage()
    print(self.phi1)
    if message == sim.message_keypress:
        #Bot movement
        if data[0] == 2007:
            self.key1 = "forward" #up arrow
        if data[0] == 2008:
            self.key1 = "backward" #down arrow
        if data[0] == 2009:
            self.key1 = "right" #left arrow
        if data[0] == 2010:
            self.key1 = "left" #right arrow
        
        #Gripper Movement
        if data[0] == 113:
            self.in_count +=1
            self.key2 = "gclose" #q
        if data[0] == 101:
            self.out_count +=1
            self.key2 = "gopen" #e
        if data[0] == 119:
            self.up_count +=1
            self.key2 = "gup" #w
        if data[0] == 115:
            self.down_count +=1
            self.key2 = "gdown" #s
    else:
        pass


def sysCall_cleanup():
    # do some clean-up here
    # This function will be executed when the simulation ends
    
    ####### ADD YOUR CODE HERE ######
    # Any cleanup (if required) to take the scene back to it's original state after simulation
    # It helps in case simulation fails in an unwanted state.
    #################################
    pass

# See the user manual or the available code snippets for additional callback functions?and?details
