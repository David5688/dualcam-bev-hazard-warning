"""Same-scene FOV/resolution demo VIDEO (blind-zone masked / clean_pred).
All frames: front/rear RGB + GT|90|120|120hr BEV, predictions blanked outside FOV.
Writes mp4v then converts to H.264 with bundled ffmpeg. Also dumps clean PNGs.
"""
import os, sys, cv2, glob, subprocess, numpy as np, torch
sys.path.insert(0, "E:/dualcam_project/src")
from dualcam_bev_lss import DualCamBEVLSS

ROOT="E:/dualcam_project"; DATA=f"{ROOT}/data"; CK=f"{ROOT}/checkpoints"
OUT=f"{ROOT}/results/samescene_viz"; os.makedirs(OUT, exist_ok=True)
FFMPEG="E:/miniforge3/envs/lss/lib/site-packages/imageio_ffmpeg/binaries/ffmpeg-win-x86_64-v7.1.exe"
dev="cuda" if torch.cuda.is_available() else "cpu"
COLOR={0:(0,0,0),1:(0,0,255),2:(255,0,255),3:(80,80,80),4:(0,255,0),5:(0,255,255)}
BEV=240; RES,X0,Y0,D0,D1=0.25,-30.0,-30.0,2.0,42.0
PNG_PICKS={"000260","000255","000265","000250","000100","000085"}

def colorize(l):
    o=np.zeros((*l.shape,3),np.uint8)
    for k,c in COLOR.items(): o[l==k]=c
    return o
def fov_mask(fov):
    r=np.arange(BEV)[:,None]; c=np.arange(BEV)[None,:]
    X=np.broadcast_to(X0+(BEV-1-r)*RES,(BEV,BEV)); Y=np.broadcast_to(Y0+c*RES,(BEV,BEV)); t=np.tan(np.radians(fov/2))
    fx,fy=X-1.5,Y; f=(fx>0)&(np.abs(fy)<=np.abs(fx)*t)&(np.hypot(fx,fy)>=D0)&(np.hypot(fx,fy)<=D1)
    rx,ry=X+1.5,Y; b=(rx<0)&(np.abs(ry)<=np.abs(rx)*t)&(np.hypot(rx,ry)>=D0)&(np.hypot(rx,ry)<=D1)
    return f|b
MASK={90:fov_mask(90),120:fov_mask(120)}; CNT={}
for fv in (90,120):
    m=(MASK[fv].astype(np.uint8)*255); m=cv2.resize(m,(480,480),interpolation=cv2.INTER_NEAREST)
    CNT[fv],_=cv2.findContours(m,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

def load(d,ck,ih,iw):
    meta=torch.load(ck,map_location="cpu",weights_only=False); ds=meta.get("lss_downsample",8); pl=meta.get("lss_pool",0.5)
    m=DualCamBEVLSS.from_calib_file(f"{d}/calibration.json",num_classes=6,bev_h=240,bev_w=240,img_h=ih,img_w=iw,downsample=ds,pool_res=pl,pretrained=False)
    st=meta["model_state_dict"] if isinstance(meta,dict) and "model_state_dict" in meta else meta
    m.load_state_dict(st,strict=True); return m.to(dev).eval()
def read(p,ih,iw):
    im=cv2.imread(p); im=cv2.resize(im,(iw,ih))[:,:,::-1].copy().astype(np.float32)/255.0
    return torch.from_numpy(im).permute(2,0,1)
def predict(m,d,n,ih,iw):
    x=torch.stack([read(f"{d}/front_rgb/{n}",ih,iw),read(f"{d}/rear_rgb/{n}",ih,iw)],0).unsqueeze(0).to(dev)
    with torch.no_grad(): return torch.argmax(m(x)[0],0).cpu().numpy().astype(np.uint8)

D90=f"{DATA}/carla_dualcam_t01_samescene_90"; D120=f"{DATA}/carla_dualcam_t01_samescene_120"; D120HR=f"{DATA}/carla_dualcam_t01_samescene_120hr"
M=[("90 standard",D90,f"{CK}/dualcam_bev_lss_m500_90_best.pt",224,400,90),
   ("120 standard",D120,f"{CK}/dualcam_bev_lss_m500_120_best.pt",224,400,120),
   ("120 HIGH-RES",D120HR,f"{CK}/dualcam_bev_lss_m500_120hr_best.pt",448,800,120)]
models=[(lbl,load(d,ck,ih,iw),d,ih,iw,fov) for lbl,d,ck,ih,iw,fov in M]

def bev_panel(color,title,fov=None,clean=True):
    P=480; v=cv2.resize(color,(P,P),interpolation=cv2.INTER_NEAREST)
    if fov is not None:
        cv2.drawContours(v,CNT[fov],-1,(255,255,255),1)  # FOV boundary
    cv2.circle(v,(P//2,P//2),5,(255,255,255),-1)
    cv2.putText(v,"FRONT",(P//2-34,18),cv2.FONT_HERSHEY_SIMPLEX,0.5,(200,200,200),1)
    cv2.rectangle(v,(0,0),(P-1,P-1),(120,120,120),1)
    bar=np.zeros((30,P,3),np.uint8); cv2.putText(bar,title,(8,21),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),1)
    return np.vstack([bar,v])

def compose(n):
    fr=cv2.resize(cv2.imread(f"{D120HR}/front_rgb/{n}"),(480,270)); re=cv2.resize(cv2.imread(f"{D120HR}/rear_rgb/{n}"),(480,270))
    cv2.putText(fr,"FRONT 120",(8,22),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,255),2)
    cv2.putText(re,"REAR 120",(8,22),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,255),2)
    inrow=np.hstack([fr,re])
    gt=cv2.imread(f"{D120}/bev_gt/{n}",cv2.IMREAD_GRAYSCALE)
    panels=[bev_panel(colorize(gt),"Ground Truth (full)",None)]
    for lbl,m,d,ih,iw,fov in models:
        p=predict(m,d,n,ih,iw); disp=colorize(p); disp[~MASK[fov]]=0   # clean: blank blind zone
        panels.append(bev_panel(disp,lbl+" (in-FOV only)",fov))
    bevrow=np.hstack(panels); W=bevrow.shape[1]
    pad=(W-inrow.shape[1])//2; inrow_c=cv2.copyMakeBorder(inrow,0,0,pad,W-inrow.shape[1]-pad,cv2.BORDER_CONSTANT,value=0)
    cap=np.zeros((26,W,3),np.uint8); cv2.putText(cap,f"same-scene Town01  frame {n}   (GT identical; only camera FOV/resolution differ)",(10,18),cv2.FONT_HERSHEY_SIMPLEX,0.5,(180,180,180),1)
    return np.vstack([inrow_c,np.zeros((8,W,3),np.uint8),bevrow,cap])

files=sorted(os.path.basename(p) for p in glob.glob(f"{D120}/bev_gt/*.png"))
h,w=compose(files[0]).shape[:2]
tmp=f"{OUT}/_tmp_mp4v.mp4"; vw=cv2.VideoWriter(tmp,cv2.VideoWriter_fourcc(*"mp4v"),15,(w,h))
for i,n in enumerate(files):
    fr=compose(n); vw.write(fr)
    if n.replace(".png","") in PNG_PICKS: cv2.imwrite(f"{OUT}/{n.replace('.png','')}_clean.png",fr)
    if i%50==0: print(f"{i}/{len(files)}",flush=True)
vw.release(); print("mp4v written, converting to H.264...")
final=f"{ROOT}/results/samescene_FOV_resolution_demo.mp4"
subprocess.run([FFMPEG,"-y","-i",tmp,"-c:v","libx264","-pix_fmt","yuv420p","-crf","20",final],check=True,capture_output=True)
os.remove(tmp)
print("DONE:",final)
