"""Same-scene FOV/resolution comparison in the PROFESSOR style (like slide_video.mp4):
black bg, gray road + green contour, yellow lanes, red vehicles, and the BLIND ZONE
drawn as a dimmed (45%) wedge -- per each model's own FOV.
Panels: Front/Rear RGB (top) + GT | 90 | 120 | 120hr BEV (bottom). Video + PNGs, H.264.
"""
import os, sys, cv2, glob, subprocess, numpy as np, torch
sys.path.insert(0, "E:/dualcam_project/src")
from dualcam_bev_lss import DualCamBEVLSS
from render_prof_style import render_bev_prof

ROOT="E:/dualcam_project"; DATA=f"{ROOT}/data"; CK=f"{ROOT}/checkpoints"
OUT=f"{ROOT}/results/samescene_prof"; os.makedirs(OUT, exist_ok=True)
FFMPEG="E:/miniforge3/envs/lss/lib/site-packages/imageio_ffmpeg/binaries/ffmpeg-win-x86_64-v7.1.exe"
dev="cuda" if torch.cuda.is_available() else "cpu"
BEV=240; RES,X0,Y0,D0,D1=0.25,-30.0,-30.0,2.0,42.0
PNG_PICKS={"000260","000255","000250","000100","000085","000180"}

def blind_of(fov):
    r=np.arange(BEV)[:,None]; c=np.arange(BEV)[None,:]
    X=np.broadcast_to(X0+(BEV-1-r)*RES,(BEV,BEV)); Y=np.broadcast_to(Y0+c*RES,(BEV,BEV)); t=np.tan(np.radians(fov/2))
    fx,fy=X-1.5,Y; f=(fx>0)&(np.abs(fy)<=np.abs(fx)*t)&(np.hypot(fx,fy)>=D0)&(np.hypot(fx,fy)<=D1)
    rx,ry=X+1.5,Y; b=(rx<0)&(np.abs(ry)<=np.abs(rx)*t)&(np.hypot(rx,ry)>=D0)&(np.hypot(rx,ry)<=D1)
    return ~(f|b)
BLIND={90:blind_of(90),120:blind_of(120)}

def smooth_road_label(lab, road_close=21, road_min=120, eps=2.5, veh_min=8):
    """Make the prediction look GT-like: road as straight-edged filled polygons,
    vehicles as clean rectangles. Lane/pedestrian kept as-is."""
    # road: close gaps, drop islands, fill holes, straighten edges (polygon approx)
    road = ((lab == 3).astype(np.uint8)) * 255
    road = cv2.morphologyEx(road, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (road_close, road_close)))
    cnts, _ = cv2.findContours(road, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rfilled = np.zeros_like(road)
    for c in cnts:
        if cv2.contourArea(c) >= road_min:
            approx = cv2.approxPolyDP(c, eps, True)       # straighten wavy edges
            cv2.drawContours(rfilled, [approx], -1, 255, -1)
    # vehicles: replace each blob with its min-area rectangle (clean boxes like GT)
    veh = ((lab == 1).astype(np.uint8)) * 255
    n, comp, stats, _ = cv2.connectedComponentsWithStats(veh, 8)
    vfilled = np.zeros_like(veh)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= veh_min:
            pts = cv2.findNonZero((comp == i).astype(np.uint8))
            box = cv2.boxPoints(cv2.minAreaRect(pts)).astype(np.int32)
            cv2.drawContours(vfilled, [box], -1, 255, -1)
    out = np.zeros_like(lab)
    out[rfilled.astype(bool)] = 3        # road base
    out[lab == 5] = 5                    # lane markings on road (thin, keep raw)
    out[lab == 2] = 2                    # pedestrians
    out[vfilled.astype(bool)] = 1        # rectangular vehicles on top
    return out

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

D90=f"{DATA}/carla_dualcam_t01_ss_90hr"; D120=f"{DATA}/carla_dualcam_t01_ss_120hr"; D120HR=f"{DATA}/carla_dualcam_t01_ss_120hr"
M=[("90 standard",D90,f"{CK}/dualcam_bev_lss_m500_90_best.pt",224,400,90),
   ("120 standard",D120,f"{CK}/dualcam_bev_lss_m500_120_best.pt",224,400,120),
   ("120 HIGH-RES",D120HR,f"{CK}/dualcam_bev_lss_m500_120hr_best.pt",448,800,120)]
models=[(lbl,load(d,ck,ih,iw),d,ih,iw,fov) for lbl,d,ck,ih,iw,fov in M]
YELLOW=(0,255,255)

def titled(bev,title,sub=None):
    P=bev.shape[0]; bar=np.zeros((34,P,3),np.uint8)
    cv2.putText(bar,title,(8,23),cv2.FONT_HERSHEY_SIMPLEX,0.62,(255,255,255),2)
    if sub: cv2.putText(bar,sub,(P-140,15),cv2.FONT_HERSHEY_SIMPLEX,0.4,(150,150,150),1)
    return np.vstack([bar,bev])

def compose(n):
    fr=cv2.resize(cv2.imread(f"{D120HR}/front_rgb/{n}"),(520,293)); re=cv2.resize(cv2.imread(f"{D120HR}/rear_rgb/{n}"),(520,293))
    cv2.putText(fr,"Front RGB",(10,26),cv2.FONT_HERSHEY_SIMPLEX,0.7,YELLOW,2)
    cv2.putText(re,"Rear RGB",(10,26),cv2.FONT_HERSHEY_SIMPLEX,0.7,YELLOW,2)
    inrow=np.hstack([fr,re])
    gt=cv2.imread(f"{D120}/bev_gt/{n}",cv2.IMREAD_GRAYSCALE)
    P=490
    panels=[titled(render_bev_prof(gt,out_size=P),"Ground Truth","full scene")]
    for lbl,m,d,ih,iw,fov in models:
        p=smooth_road_label(predict(m,d,n,ih,iw))
        bev=render_bev_prof(p,out_size=P,dim_blind=BLIND[fov],smooth_road=False,min_vehicle=15,min_lane=15)
        panels.append(titled(bev,lbl,f"blind {int(100*BLIND[fov].mean())}%"))
    bevrow=np.hstack(panels); W=bevrow.shape[1]
    pad=(W-inrow.shape[1])//2; inrow_c=cv2.copyMakeBorder(inrow,0,0,pad,W-inrow.shape[1]-pad,cv2.BORDER_CONSTANT,value=0)
    cap=np.zeros((30,W,3),np.uint8)
    cv2.putText(cap,f"Same-scene Town01  frame {n}   -   blind zone dimmed to 45% per camera FOV   -   GT identical, only FOV/resolution differ",
                (12,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(180,180,180),1)
    return np.vstack([inrow_c,np.zeros((8,W,3),np.uint8),bevrow,cap])

files=sorted(os.path.basename(p) for p in glob.glob(f"{D120}/bev_gt/*.png"))
h,w=compose(files[0]).shape[:2]
tmp=f"{OUT}/_tmp.mp4"; vw=cv2.VideoWriter(tmp,cv2.VideoWriter_fourcc(*"mp4v"),12,(w,h))
for i,n in enumerate(files):
    fr=compose(n); vw.write(fr)
    if n.replace(".png","") in PNG_PICKS: cv2.imwrite(f"{OUT}/{n.replace('.png','')}_prof.png",fr)
    if i%50==0: print(f"{i}/{len(files)}",flush=True)
vw.release(); print("converting H.264...")
final=f"{ROOT}/results/samescene_FOV_prof_style.mp4"
subprocess.run([FFMPEG,"-y","-i",tmp,"-c:v","libx264","-pix_fmt","yuv420p","-crf","20",final],check=True,capture_output=True)
os.remove(tmp); print("DONE:",final,f"({w}x{h})")
