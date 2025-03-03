import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
import pyzed.sl as sl
import time
import math
import Kalman_filterlib as KF
import utils as Uts

x=0
pt=0
pttt=0
ptttinT=[]
ptinT=[]
init_params=sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.HD720
init_params.camera_fps = 15
zed=sl.Camera() 
status1=zed.open(init_params)
mat1=sl.Mat()
mat2=sl.Mat()			   
win_name2="live CAM"       # Right cam(according to calibration)    Left cam(according to calibration)
win_name="match Live Cam"          # fx=524.301 cx=672.224            fx=524.284  cx=680.445     
cv.namedWindow(win_name)	   # fy=523.312  cy=367.930            fy=524.740   cy=370.417
cv.namedWindow(win_name2)
Key=''
runtime=sl.RuntimeParameters()
sensors_data=sl.SensorsData()
K=np.array([[524.284,0,680.445],
			 [0,524.740,370.417],
			 [0,0,1]])
KR=np.array([[524.301,0,672.224 ],
			 [0,523.312 ,367.930],
			 [0,0,1]])
P=np.array([[524.301,0,672.224 ,0],[0,523.312,367.930,0],[0,0,1,0]])
PL=np.array([[524.284,0,680.445 ,0],
			 [0,524.740,370.417,0],
			 [0,0,1,0]])
PR=np.array([[524.301,0,672.224 ,0],
			 [0,523.312,367.930,0],
			 [0,0,1,0]])
baseline = -0.012  # -12 cm baseline (negative)
R_right = np.eye(3)
t_right = np.array([[baseline], [0], [0]], dtype=float)
PRR= np.dot(KR, np.hstack((R_right, t_right)))
Ident=np.identity(3)
timedif=0
sampling=0
sampling_minus1=0
x=0
Ciclos=0
g=9.81

"""Rt=np.array([[0.011,0,0],
		[0,0.011,0],
		[0,0,0.011]])"""
Rt2=np.array([[0.011,0,0,0],
		[0,0.011,0,0],
		[0,0,0.011,0],
		[0,0,0,0.011]])
Qt2=np.array([[0.001,0,0],
		[0,0.001,0],
		[0,0,0.001]])
"""Qt=np.array([[0.001,0],
		[0,0.001]])"""
phi=0
theta=0
psi=0
theta_dot=0
phi_dot=0
psi_dot=0
p=0
q=0
r=0
ax=0
ay=0
az=0
Identity=np.array([[1,0],
			[0,1]])
Pt2_minus11=np.array([[0.1,0,0],
			[0,0.1,0],
			[0,0,0.1]])
Dt1=0.03
Dt2=0.1


def PredictionV(Xkm,Uk,Dt,Pkm,Qk):
	Xk=Xkm+0.1*(Xkm+Uk*Dt)
	print('Xk',Xk)
	Pk=Pkm+Qk
	return Xk,Pk

def upgradeV(Zk,Xk,Pk,Rk):
	Zmedia=Zk-Xk
	Sk=Pk+Rk
	Kk=Pk/Sk
	Xk=Xk+Kk*Zmedia
	Pk=(1-Kk)*Pk
	return Xk,Pk

X=0
V=0
if __name__=='__main__':
	A1 = np.array([[1, Dt2],
					  [0, 1]])
	C1 = np.array([[1, 0], [1, 0]])
	Identity=np.array([[1,0,0],
			[0,1,0],
			[0,0,1]])
	Identity2=np.array([[1,0],
			[0,1]])
	U_k1=np.array([0,0])
	U_k2=np.array([0,0])
	max_depth=6.3
	min_depth=0.2
	min_disp=13
	max_disp=50
	Aprox_Vel=0
	Aprox_VelN=0
	Aprox_VelIMU=0
	Co=0
	magT=[]
	locT=[]
	mag_heading2=0
	AccinT=[]
	Aprox_VelinT=[]
	Aprox_VelIMUinT=[]
	initial_state = np.array([[0], [0]])  # Initial state: [angular position, angular velocity]
	initial_covariance = np.eye(2)  # Initial covariance matrix Pk
	
	process_noise = np.diag([0.5, 0.5])  # Process noise covariance matrix Qk
	
	measurement_noise = np.diag([0.6, 0.6])  # Measurement noise covariance matrix Rk
	
	
	Pk=0.5
	Qk=0.9
	Rk=0.6

	Px=0
	Py=0
	VxinT=[]
	VyinT=[]
	PxinT=[]
	PyinT=[]
	theta_xinT=[]
	theta_yinT=[]
	theta_zinT=[]
	root=os.getcwd()
	start_pose=np.ones((3,4))
	start_translation=np.zeros((3,1))
	start_rotation=np.identity(3)
	start_pose=np.concatenate((start_rotation,start_translation),axis=1)
	if status1 != sl.ERROR_CODE.SUCCESS:
		print("Camera Open : "+repr(status1)+". Exit program.")
		exit()
	TT=mat1.timestamp.get_microseconds()
	print("TT")
	print(TT)
	timepassed=time.time()
	seconds_passed=time.time()
	alphaV=0.9
	alphaA=0.9
	alphaVL=0.92
	Update_steps=0
	IMU=np.array([0,0,0])
	KF1=KF.KalmanFilter(dt=Dt1,P_k=Pt2_minus11,Qt=Qt2,Rt=Rt2,Xk=IMU,I=Identity)
	KF2=KF.KalmanFilter(dt=Dt2,P_k=initial_covariance,Qt=process_noise,Rt=measurement_noise,Xk=initial_state,I=Identity2)
	KF3=KF.KalmanFilter(dt=Dt2,P_k=initial_covariance,Qt=process_noise,Rt=measurement_noise,Xk=initial_state,I=Identity2)
	IMU2=np.array([0,0,0])
	IMU_dot=np.array([0,0,0])
	A_velocityN=np.array([0,0,0])
	AcceleretionN=np.array([0,0,0])
	loc_headingN=0
	theta_xN=0
	theta_yN=0
	theta_zN=0
	NotF_in_time=[]
	F_in_time=[]
	TIMEforG=[]
	Pitch_inTIME=[]
	Yaw_inTIME=[]
	RollinTIME=[]
	yaw_yawclass=[]
	TIMEforP=[]
	RAWG=np.array([0,0,0])
	RAWGinT=[]
	SecKFinT=[]
	ThirdKFinT=[]
	dp=Uts.DepthMap(showImages=False)
	utils=Uts.utils(K,P,Ident,PL,PRR)
	
	
	if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
		zed.retrieve_image(mat1,sl.VIEW.LEFT)
		cv.imwrite('images//savedorbT1_l.png',mat1.get_data())
		zed.retrieve_image(mat2,sl.VIEW.RIGHT)
		cv.imwrite('images//savedorbT1_R.png',mat2.get_data())
		imgPath2l=os.path.join(root,'images//savedorbT1_l.png')
		imgPath2r=os.path.join(root,'images//savedorbT1_R.png')
		cvImage_minus1_L=cv.imread(imgPath2l)
		cvImage_minus1_R=cv.imread(imgPath2r)
		disparity_m1=dp.computeDepthMapSGBM(cvImage_minus1_L,cvImage_minus1_R,showImages=False)
		print('punto__________')
	TM=start_pose
	seconds=time.time()
	seconds2=time.time()
	while Key!=113:
		Tat=math.tan(theta)
		Cp=math.cos(phi)
		Ct=math.cos(theta)
		Sp=math.sin(phi)
		St=math.sin(theta)
		err=zed.grab(runtime)
		Fdxm=np.array([[1,Sp*Tat,Cp*Tat],
		 [0,Cp,-Sp],
		 [0,(Sp/Ct),(Cp/Ct)]])
		Hdxm=np.array(
		[g*St,-g*Ct*Sp,-g*Ct*Cp,psi])
		#A=np.array([[q*Cp*Tat-r*Sp*Tat,r*Cp*(Tat*Tat+1)+q*Sp*(Tat*Tat+1)],[-r*Cp-q*Sp,0]]) ###Cambiando las matrices del kalman para incluir magnetometro
		A=np.array([[q*Cp*Tat-r*Sp*Tat,r*Cp*(Tat*Tat+1)+q*Sp*(Tat*Tat+1),0],
		[-r*Cp-q*Sp,0,0],
        [(q*Cp)/Ct-(r*Sp)/Ct,(r*Cp*St)/(Ct*Ct)+(q*Sp*St)/(Ct*Ct), 0]])
		#C=np.array([[0,g*Ct],[-g*Cp*Ct,g*Sp*St],[g*Sp*Ct,g*Cp*St]])
		C=np.array([[0,g*Ct, 0],
		[-g*Cp*Ct, g*Sp*St,0],
		[ g*Ct*Sp, g*Cp*St,0],
		[ 0,0, 1]])
		sampling_minus1=time.time()
		

		if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
			
			zed.retrieve_image(mat2,sl.VIEW.RIGHT)
			cvImage_R=mat2.get_data()
			zed.retrieve_image(mat1,sl.VIEW.LEFT)
			cvImage_L=mat1.get_data()
			timePassed3=time.time()-seconds
			if err == sl.ERROR_CODE.SUCCESS:
				zed.get_sensors_data(sensors_data, sl.TIME_REFERENCE.IMAGE)
				#zed.retrieve_image(mat1,sl.VIEW.RIGHT)
				if mat1.timestamp.get_microseconds() > TT:

					timedif=time.time()-seconds_passed
					imu_data=sensors_data.get_imu_data()
					linear_acceleration=imu_data.get_linear_acceleration()
					angular_velocity=imu_data.get_angular_velocity()
					magnetometer_data = sensors_data.get_magnetometer_data()
					magnetic_field_dat=magnetometer_data.get_magnetic_field_calibrated()
					mag_heading=np.arctan2(magnetic_field_dat[0],magnetic_field_dat[2])
					mag_heading=mag_heading
					if mag_heading<0:
						mag_heading=mag_heading+2*np.pi
					if Co<2:
						#mag_heading2=np.arctan(magnetic_field_dat[0]/magnetic_field_dat[2])*180/np.pi
						mag_heading2=np.arctan2(magnetic_field_dat[0],magnetic_field_dat[2])
						mag_heading2=mag_heading2
						if mag_heading2<0:
							mag_heading2=mag_heading2+2*np.pi
						Co+=1
					loc_heading=mag_heading-mag_heading2
					loc_headingN=0.7*loc_headingN+(1-0.7)*loc_heading
					p=-math.radians(angular_velocity[2])
					q=-math.radians(angular_velocity[0]) #remapped
					r=math.radians(angular_velocity[1])
					A_velocity=np.array([p,q,r])
					A_velocityN=alphaV*A_velocityN+(1-alphaV)*A_velocity
					ax=-linear_acceleration[2]
					ay=-linear_acceleration[0]  #remapped
					az=linear_acceleration[1]
					Acceleretion=np.array([ax,ay,az])
					AcceleretionN=alphaA*AcceleretionN+(1-alphaA)*Acceleretion
					
					KF1.predict(U_k=A_velocityN,C=C,A=A,fd=Fdxm)                   #predict stage
					IMU=KF1.X_k
					phi=IMU[0]
					theta=IMU[1]
					psi=IMU[2]
					x=x+1
					Ciclos=x/timedif
					TT=mat1.timestamp.get_microseconds()	

					NotF_in_time.append([p,q,r])
					F_in_time.append(A_velocityN)
					TIMEforG.append(timedif)

					sampling=time.time()-sampling_minus1
					print(f'TIMESampling:{sampling}')
					Time_reference=time.time()-timepassed 
					if Time_reference > 0.1:
						X=X+1
						cv.imwrite('images//savedorbT_m1_l.png', cvImage_minus1_L)
						cv.imwrite('images//savedorbT_m1_R.png', cvImage_minus1_R)
						cv.imwrite('images//savedorbT1_l.png', cvImage_L)
						cv.imwrite('images//savedorbT1_R.png', cvImage_R)
						disparityN=dp.computeDepthMapSGBM(cvImage_L,cvImage_R,showImages=False)
						
						frame,TMn,srcPts,dstPts,goodMatches=utils.ORBDiferent(cvImage_L,cvImage_minus1_L)
						
						seconds=time.time()
						V=1
						
						print(X)
						Zk1=np.append(AcceleretionN,loc_headingN)
						KF1.update(Zk=Zk1,C=C,Hd=Hdxm) #update stage
						
						Update_steps+=1
						print(f'update:{Update_steps}')
						IMU=KF1.X_k
						
						timepassed1=time.time()-seconds_passed 
						timepassed=time.time()
						Pitch_inTIME.append(math.degrees(IMU[0]))
						Yaw_inTIME.append(math.degrees(IMU[1]))
						RollinTIME.append(math.degrees(IMU[2]))
						TIMEforP.append(timepassed1)
						RAWG=RAWG+A_velocityN*0.3*180/np.pi
						RAWGinT.append(RAWG)
					
						##############################
						TM=TM @ TMn
						R_n=TM[:3, :3]
						theta_x, theta_y, theta_z=utils.TtoEuler(R_n)
						theta_xN=0.3*theta_xN+(1-0.3)*theta_x
						theta_yN=0.3*theta_yN+(1-0.3)*theta_y   #loc_headingN=0.7*loc_headingN+(1-0.7)*loc_heading                      
						theta_zN=0.3*theta_zN+(1-0.3)*theta_z
						theta_xinT.append(theta_xN)
						theta_yinT.append(theta_yN)
						theta_zinT.append(theta_zN)
						magT.append(mag_heading)
						locT.append(IMU[2])
						KF2.predict(U_k=U_k1,C=C1,A=A1)
						KF2.update(Zk=np.array([[theta_xN], [math.degrees(IMU[1])]]),C=C1) 
						U_k1=KF2.X_k
						KF3.predict(U_k=U_k2,C=C1,A=A1)
						KF3.update(Zk=np.array([[theta_yN], [math.degrees(IMU[2])]]),C=C1) 
						U_k2=KF3.X_k
						depth_values1,depth_values2,disparityN1,disparityN2=utils.Depth_for_points(srcPts,dstPts,disparity_m1,disparityN,goodMatches)
						
						
						ThirdKFinT.append(U_k2[1])
						SecKFinT.append(U_k1[1])

					if V==1:
						frame=cv.resize(frame,(500,750))
						cv.imshow(win_name, frame)
						V=0

						disparityN1,left_x11,left_y11,rightx11=utils.delete_outliersD(srcPts,disparityN1,max_disp,min_disp)
						disparityN2,left_x22,left_y22,rightx22=utils.delete_outliersD(dstPts,disparityN2,max_disp,min_disp)


						q1_l=np.column_stack((left_x11, left_y11))
						q1_r=np.column_stack((rightx11,left_y11))
						q2_l=np.column_stack((left_x22,left_y22))
						q2_r=np.column_stack((rightx22,left_y22))

						Q1,Q2=utils.triangulate(q1_r,q1_l,q2_r,q2_l)

						Aprox_Vel=(np.mean(Q2[:,2])-np.mean(Q1[:,2]))/0.1
						print(Aprox_Vel)
						Aprox_VelN=alphaVL*Aprox_VelN+(1-alphaVL)*Aprox_Vel
						Aprox_VelinT.append(Aprox_VelN)

						disparity_m1=disparityN

						Pkm=Pk
						print('PKM:',Pkm)
						Dt=0.1
						AccinT.append(AcceleretionN[0])
						Xk,Pk=PredictionV(Aprox_VelIMU,AcceleretionN[0],Dt,Pkm,Qk)    ###MODIFICAR VARIABLES
						if X>1:
							Xk,Pk=upgradeV(Aprox_VelN,Xk,Pk,Rk)
							X=0
							Aprox_VelIMU=Xk
						Vy=Aprox_VelIMU*math.cos(math.radians(loc_headingN))
						Vx=Aprox_VelIMU*math.sin(math.radians(loc_headingN))
						if abs(A_velocityN[2])>0.01:
							Vx=0
							Vy=0
						VxinT.append(Vx)
						VyinT.append(Vy)
						Px=(Px+Vx*Dt)
						Py=(Py+Vy*Dt)
						PxinT.append(Px)
						PyinT.append(Py)
						Aprox_VelIMUinT.append(Aprox_VelIMU)

					

						

					elif(V==0):
						#print(timePassed)
						imgPath2r=os.path.join(root,'images//savedorbT1_r.png')
						imgPath2l=os.path.join(root,'images//savedorbT1_l.png')
						cvImage_minus1_L=cv.imread(imgPath2l)
						cvImage_minus1_R=cv.imread(imgPath2r)
						
						
				
		else:
			print("Error during capture:", err)
			break
		cv.putText(cvImage_L, str(np.round(TM[0, 0],2)), (260,50), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)
		cv.putText(cvImage_L, str(np.round(TM[0, 1],2)), (340,50), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)
		cv.putText(cvImage_L, str(np.round(TM[0, 2],2)), (420,50), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)
		cv.putText(cvImage_L, str(np.round(TM[1, 0],2)), (260,90), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)
		cv.putText(cvImage_L, str(np.round(TM[1, 1],2)), (340,90), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)
		cv.putText(cvImage_L, str(np.round(TM[1, 2],2)), (420,90), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)
		cv.putText(cvImage_L, str(np.round(TM[2, 0],2)), (260,130), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)
		cv.putText(cvImage_L, str(np.round(TM[2, 1],2)), (340,130), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)
		cv.putText(cvImage_L, str(np.round(TM[2, 2],2)), (420,130), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)
		cv.putText(cvImage_L, str(np.round(TM[0, 3],2)), (540,50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
		cv.putText(cvImage_L, str(np.round(TM[1, 3],2)), (540,90), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
		cv.putText(cvImage_L, str(np.round(TM[2, 3],2)), (540,130), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
		cv.putText(cvImage_L, str(np.round(Aprox_VelN,2)), (640,230), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
		cv.imshow(win_name2, cvImage_L)
		Key=cv.waitKey(5)  
	cv.destroyAllWindows()
	zed.close()
	NotF_in_time=np.array(NotF_in_time)
	F_in_time=np.array(F_in_time)
	TIMEforG=np.array(TIMEforG)
	RAWGinT=np.array(RAWGinT)
	Yy1=NotF_in_time[:,1]
	Yy2=F_in_time[:,1]
   
	# filtered Vs Not filtered


	plt.figure(1)
	plt.plot(TIMEforP,Yaw_inTIME,'r',label='filtered')
	plt.plot(TIMEforP,RAWGinT[:,1],'b',label='not filtered')
	plt.title("angle vs Time")
	plt.xlabel("Time(s)")
	plt.ylabel("Angle(grad)")
	plt.xticks(fontsize = 15) 
	plt.yticks(fontsize = 15) 
	plt.legend(prop={'size': 15})
	plt.show()

	plt.figure(2)
	plt.plot(TIMEforP,magT,'r',label='mag_heading')
	plt.plot(TIMEforP,locT,'b',label='Loc_heading')
	plt.title("angle vs Time")
	plt.xlabel("Time(s)")
	plt.ylabel("Angle(grad)")
	plt.xticks(fontsize = 15) 
	plt.yticks(fontsize = 15) 
	plt.legend(prop={'size': 15})
	plt.show()

#Visual Vs Not Visual
	plt.figure(3)
	plt.plot(TIMEforP,theta_xinT,'r',label='V odometry') #Pitch_inTIME Yaw_inTIME
	plt.plot(TIMEforP,Yaw_inTIME,'b',label='filtered')
	plt.title("angle vs Time")
	plt.xlabel("Time(s)")
	plt.ylabel("Angle(grad)")
	plt.xticks(fontsize = 15) 
	plt.yticks(fontsize = 15) 
	plt.legend(prop={'size': 15})
	plt.show()
	print(len(TIMEforP))
	
	plt.figure(4)
	plt.plot(TIMEforP,theta_xinT,'r',label='V odometry')
	plt.plot(TIMEforP,SecKFinT,'b',label='second Kalman')
	plt.title("angle vs Time")
	plt.xlabel("Time(s)")
	plt.ylabel("Angle(grad)")
	plt.xticks(fontsize = 15) 
	plt.yticks(fontsize = 15) 
	plt.legend(prop={'size': 15})
	plt.show() 

	plt.figure(5)
	plt.plot(TIMEforP,theta_yinT,'r',label='VO heading')
	plt.plot(TIMEforP,RollinTIME,'b',label='Loc_heading')
	plt.plot(TIMEforP,ThirdKFinT,'g',label='filtered heading')
	plt.title("angle vs Time")
	plt.xlabel("Time(s)")
	plt.ylabel("Angle(grad)")
	plt.legend(prop={'size': 15})
	plt.show() 
	#Linear velocity 
	plt.figure(6) 
	plt.plot(TIMEforP,Aprox_VelinT,'r',label='V linear odometry')
	plt.plot(TIMEforP,Aprox_VelIMUinT,'b',label='V linear')
	plt.title("Velocity vs Time")
	plt.xlabel("Time(s)")
	plt.ylabel("M/s")
	plt.legend(prop={'size': 15})
	plt.show() 
	
	#plt.figure(5) 
	#plt.plot(TIMEforP,AccinT,'r',label='Acceleration')
	#plt.title("acceleration vs Time")
	#plt.xlabel("Time(s)")
	#plt.ylabel("M/s/s")
	#plt.legend(prop={'size': 15})
	
	#linear pose 

	plt.figure(7)
	plt.plot(PxinT,PyinT,'r',label='Pose')
	plt.title("Position X vs Position Y")
	plt.xlabel("Position X")
	plt.ylabel("Position Y")
	plt.legend(prop={'size': 15})
	plt.show() 
 
