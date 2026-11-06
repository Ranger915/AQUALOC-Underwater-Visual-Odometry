from common import *
rng=np.random.default_rng(20261105);X=np.column_stack([rng.uniform(-5,5,100),rng.uniform(-5,5,100),rng.uniform(100,150,100)]);t=np.array([1.,0,0]);Y=X+t
def project(z):
 a=z@K.T;return a[:,:2]/a[:,2,None]
a=project(X);b=project(Y);E=np.array([[0,0,0],[0,0,-1],[0,1,0.]],float);mask=np.ones((100,1),np.uint8)
small=cv2.recoverPose(E,a,b,K,distanceThresh=50.,mask=mask.copy());large=cv2.recoverPose(E,a,b,K,distanceThresh=1e9,mask=mask.copy())
assert len(small)==len(large)==5 and small[0]==0 and large[0]==100
dump(OUT/'depth_overload_validation.json',dict(points=100,positive_depth_range=[100,150],baseline=1,cutoff50_inliers=int(small[0]),cutoff1e9_inliers=int(large[0]),return_values=len(large),passed=True,invocation='recoverPose(..., cameraMatrix=K positional, distanceThresh=cutoff keyword, mask=mask.copy())'))
print('Explicit distance overload validated:',small[0],large[0])
