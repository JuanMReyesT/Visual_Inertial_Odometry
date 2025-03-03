import cv2 as cv
import numpy as np
import math
import matplotlib.pyplot as plt


class DepthMap:
	def __init__(self,showImages):
		self.cccc=1
		
		if showImages:
			plt.figure()
			plt.subplot(121)
			plt.imshow(self.imgRight)
			plt.subplot(122)
			plt.imshow(self.imgLeft)
			plt.show()
	def computeDepthMapBM(self):
		nDispFactor=6
				 #to adjust
		stereo=cv.StereoBM.create(numDisparities=16*nDispFactor,blockSize=21)
		disparity=stereo.compute(self.imgLeft,self.imgRight)
		plt.imshow(disparity,'gray')
		plt.show()

	def computeDepthMapSGBM(self,imgLeftx,imgRightx,showImages):
		imgLeft=cv.cvtColor(imgLeftx,cv.COLOR_BGR2GRAY)
		imgRight=cv.cvtColor(imgRightx,cv.COLOR_BGR2GRAY)
		#imgLeft=cv.resize(imgLeftx,(520, 520))
		#imgRight=cv.resize(imgRightx,(520, 520))
		window_size=12
		Min_disp=4
		nDispFactor=10
		num_disp=16*nDispFactor-Min_disp
		if showImages:
			plt.figure()
			plt.subplot(121)
			plt.imshow(imgLeft)
			plt.subplot(122)
			plt.imshow(imgRight)
			plt.show()
		
		stereo=cv.StereoSGBM_create(minDisparity=Min_disp,numDisparities=num_disp,blockSize=window_size,P1=8*4*window_size**2,P2=32*4*window_size**2,
											disp12MaxDiff=1,uniquenessRatio=15,speckleWindowSize=0,speckleRange=2,preFilterCap=63,mode=cv.STEREO_SGBM_MODE_SGBM_3WAY)
		
		right_matcher = cv.ximgproc.createRightMatcher(stereo)

		disparity=stereo.compute(imgLeft,imgRight).astype(np.float32)/16
		
		rightD=right_matcher.compute(imgRight,imgLeft).astype(np.float32)/16


		lmbda=6000.0
		sigma=3

		wls=cv.ximgproc.createDisparityWLSFilter(stereo)

		wls.setLambda(lmbda)
		wls.setSigmaColor(sigma)

		filteredDis=wls.filter(disparity,imgLeft,disparity_map_right=rightD)

		#plt.imshow(filteredDis)
		#plt.colorbar()
		#plt.show()
		#plt.imshow(filteredDis)
		#depth=(2.12*120)/(filteredDis*0.002)
		#plt.imshow(disparity)
		#plt.colorbar()
		
		return filteredDis	
class utils:
	def __init__(self,K,P,Ident,PL,PRR):
		self.K=K
		self.Ident=Ident
		self.P=P
		self.PL=PL
		self.PRR=PRR
	def TransformMatrix(self,R,t):
		T=np.eye(4,dtype=np.float64)
		T[:3,:3]=R
		T[:3,3]=t
		return T
	def ORBDiferent(self,cvImage,cvImage_minus1):
		#root=os.getcwd()
		#imgPath1=os.path.join(root,'test_ORBImages//bureau1.jpeg')#BMW1.png bureau1.jpeg saved.jpeg
		#imgPath2=os.path.join(root,'test_ORBImages//saved2.jpeg')#BMW2.png bureau2.jpeg saved2.jpeg
		imgGray1=cv.cvtColor(cvImage_minus1,cv.COLOR_BGR2GRAY)#cv.imread(imgPath1,cv.IMREAD_GRAYSCALE)
		imgGray2=cv.cvtColor(cvImage,cv.COLOR_BGR2GRAY)#cv.imread(imgPath2,cv.IMREAD_GRAYSCALE)
		#imgGray1=cv.resize(imgGray1,(520, 520))
		#imgGray2=cv.resize(imgGray2,(520, 520))

		orb=cv.ORB_create(5000)
		FLANN_INDEX_LSH=6	
		index_params=dict(algorithm=FLANN_INDEX_LSH,table_number=6,key_size=12,multi_probe_level=1)
		search_params=dict(checks=60)
		flann=cv.FlannBasedMatcher(indexParams=index_params,searchParams=search_params)	
		
		keypoints1,descriptor1=orb.detectAndCompute(imgGray1,None)
		keypoints2,descriptor2=orb.detectAndCompute(imgGray2,None)
		matches=flann.knnMatch(descriptor1,descriptor2,k=2)
		goodMatches=[]
		print("______")
		print(len(matches))
		print("----------")
		for m,n in matches:
			if m.distance<0.8*n.distance:
				goodMatches.append(m)
		MinM=20	
		if len(goodMatches)>MinM:
			srcPts=np.float32([keypoints1[m.queryIdx].pt for m in goodMatches])#.reshape(-1,1,2)
			dstPts=np.float32([keypoints2[m.trainIdx].pt for m in goodMatches])#.reshape(-1,1,2)
			errorThreshold=6
			#print(dstPts)
			#print("_____________")
			E,maskE=cv.findEssentialMat(srcPts,dstPts,self.K,cv.RANSAC,0.999,errorThreshold)
			matchesMask=maskE.ravel().tolist()
			h,w=imgGray1.shape
			R1,R2,t=cv.decomposeEssentialMat(E)
			T1=self.TransformMatrix(R1,np.ndarray.flatten(t))
			T2=self.TransformMatrix(R2,np.ndarray.flatten(t))
			T3=self.TransformMatrix(R1,np.ndarray.flatten(-t))
			T4=self.TransformMatrix(R2,np.ndarray.flatten(-t))
			Transformations=[T1,T2,T3,T4]
			Kc=np.concatenate((self.K,np.zeros((3,1)) ),axis=1) 
			IdentC=np.concatenate((self.Ident,np.zeros((3,1)) ),axis=1)
			Projections=[Kc @ T1, Kc @ T2 , Kc @ T3  , Kc @ T4 ] #4x3
			Rs,ts,vs,_=cv.recoverPose(E,srcPts,dstPts)
			#print("recovered T")
			T0=np.concatenate((ts,vs),axis=1)
			T0=np.row_stack([T0,np.array([0,0,0,1])])
			#print(T0)
			#print("recovered T")
			positives=[]
			for Pc , T in zip(Projections,Transformations):
				HQ1=cv.triangulatePoints(self.P,Pc,srcPts.T,dstPts.T)
				HQ2=T @ HQ1
				Q1=HQ1[:3,:]/HQ1[3,:]
				Q2=HQ2[:3,:]/HQ2[3,:]
				total_sum=sum(Q2[2,:]>0)+sum(Q1[2,:]>0)
				relative_scale = np.mean(np.linalg.norm(Q1.T[:-1]-Q1.T[1:],axis=1)/np.linalg.norm(Q2.T[:-1]-Q2.T[1:],axis=1))
				positives.append(total_sum+relative_scale)
			#print(positives)
			max=np.argmax(positives)
			if(max==2):
				print("2")
				R=R1
				tt=-t
				print(R)
				print(tt)
			elif(max==3):
				print("3")
				R=R2
				tt=-t
				print(R)
				print(tt)
			elif(max==0):
				print("0")
				R=R1
				tt=t
				print(R)
				print(tt)
			elif(max==1):
				print("1")
				R=R2
				tt=t
				print(R)
				print(tt)
			DrawParams=dict(matchColor=-1,singlePointColor=None,matchesMask=matchesMask,flags=cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
			imgMatches=cv.drawMatches(imgGray1,keypoints1,imgGray2,keypoints2,goodMatches,None,**DrawParams) 
			TransformM=self.TransformMatrix(R,np.ndarray.flatten(tt))
			print("Enough matches",len(goodMatches))
		else: 
			print("Not enough matches")
			imgMatches=imgGray1	
		
		#plt.figure()
		#plt.imshow(imgMatches)
		#plt.show()
		return imgMatches,TransformM,srcPts,dstPts,goodMatches
	def TtoEuler(self,R):

		
		# Roll (x-axis rotation)
		theta_x = np.arctan2(R[2, 1], R[2, 2])*180/np.pi
			# Pitch (y-axis rotation)
		theta_y = np.arctan2(-R[2, 0], np.sqrt(R[2, 1]**2 + R[2, 2]**2))*180/np.pi
			# Yaw (z-axis rotation)
		theta_z = np.arctan2(R[1, 0], R[0, 0])*180/np.pi
		r,_=cv.Rodrigues(R)
		print('rodriguez')
		print(r*180/np.pi)
		print('rodriguez')
		print('mymethod')
		print([theta_x,theta_y,theta_z])
		print('mymethod')
		return theta_x, theta_y, theta_z
	def Depth_for_points(self,srcPts,dstPts,disparity1,disparity2,goodMatches):
		left_y1, left_x1=srcPts[:,0],srcPts[:,1]
		left_y2, left_x2=dstPts[:,0],dstPts[:,1]
		disparityN1=[]
		disparityN2=[]
		depth_values1=[]
		depth_values2=[]

		for i in range(len(goodMatches)):
			#print(i)
			#print(left_x[i])
			disparity_value1 = disparity1[int(left_x1[i]), int(left_y1[i])]
			disparity_value2 = disparity2[int(left_x2[i]), int(left_y2[i])]
			depth1 =  0.12 * 529 / disparity_value1 #depth=(2.12*120)/(filteredDis*0.002)
			depth2 =  0.12 * 529 / disparity_value2 #depth=(2.12*120)/(filteredDis*0.002)
			disparityN1.append(disparity_value1)
			disparityN2.append(disparity_value2)

			depth_values1.append(depth1)
			depth_values2.append(depth2)
		return depth_values1,depth_values2,disparityN1,disparityN2
	def delete_outliers(self,depth_values,Pts,disparityNs,max_depth,min_depth):
		left_y, left_x=Pts[:,0],Pts[:,1]
		rightx=left_x-disparityNs
		depth_valuesA= np.array(depth_values)/depth_values[3]
		
		DepthMASK1=np.where( depth_valuesA > max_depth)[0]
		
		
		depth_valuesA=np.delete(depth_valuesA,DepthMASK1)
		DepthMASK2=np.where( depth_valuesA < min_depth)[0]
		depth_valuesA=np.delete(depth_valuesA,DepthMASK2)
		NANMASK=np.isnan(depth_valuesA)
		left_xA=np.array(left_x)
		left_xA=np.delete(left_xA,DepthMASK1)
		left_xA=np.delete(left_xA,DepthMASK2)
		left_xA=left_xA[~NANMASK]
		left_yA=np.array(left_y)
		left_yA=np.delete(left_yA,DepthMASK1)
		left_yA=np.delete(left_yA,DepthMASK2)
		left_yA=left_yA[~NANMASK]
		rightxA=np.array(rightx)
		rightxA=np.delete(rightxA,DepthMASK1)
		rightxA=np.delete(rightxA,DepthMASK2)
		rightxA=rightxA[~NANMASK]
		left_x=left_xA.flatten().tolist()
		left_y=left_yA.flatten().tolist()
		rightx=rightxA.flatten().tolist()
		depth_valuesA= depth_valuesA[~NANMASK]
		depth_values=depth_valuesA.flatten().tolist()
		return depth_values,left_x,left_y,rightx
	def delete_outliersD(self,Pts,disparityNs,max_disp,min_disp):
		left_y, left_x=Pts[:,0],Pts[:,1]
		rightx=left_x-disparityNs
		disparityNsA= np.array(disparityNs)
		
		DepthMASK1=np.where( disparityNsA > max_disp)[0]
		
		
		disparityNsA=np.delete(disparityNsA,DepthMASK1)
		DepthMASK2=np.where( disparityNsA < min_disp)[0]
		disparityNsA=np.delete(disparityNsA,DepthMASK2)
		NANMASK=np.isnan(disparityNsA)
		left_xA=np.array(left_x)
		left_xA=np.delete(left_xA,DepthMASK1)
		left_xA=np.delete(left_xA,DepthMASK2)
		left_xA=left_xA[~NANMASK]
		left_yA=np.array(left_y)
		left_yA=np.delete(left_yA,DepthMASK1)
		left_yA=np.delete(left_yA,DepthMASK2)
		left_yA=left_yA[~NANMASK]
		rightxA=np.array(rightx)
		rightxA=np.delete(rightxA,DepthMASK1)
		rightxA=np.delete(rightxA,DepthMASK2)
		rightxA=rightxA[~NANMASK]
		left_x=left_xA.flatten().tolist()
		left_y=left_yA.flatten().tolist()
		rightx=rightxA.flatten().tolist()
		disparityNsA= disparityNsA[~NANMASK]
		disparityNs=disparityNsA.flatten().tolist()
		return disparityNs,left_x,left_y,rightx
	def triangulate(self,q1r,q1l,q2r,q2l):
		#sssss
		Q1=cv.triangulatePoints(self.PL,self.PRR,q1l.T,q1r.T)
		Q2=cv.triangulatePoints(self.PL,self.PRR,q2l.T,q2r.T)
		Q1 = np.transpose(Q1[:3] / Q1[3])
		Q2 = np.transpose(Q2[:3] / Q2[3])
		return Q1,Q2
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