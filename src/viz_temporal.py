"""Temporal BEV fusion demo: fill the current blind zone with STATIC road/lane/
sidewalk remembered from earlier frames (warped by ego motion), so turns stay
connected instead of fragmenting. Moving objects (vehicle/ped) use the current
frame only (no ghosting). Panels: Front/Rear RGB + Current | Temporal | GT.
"""
import os, sys, cv2, json, math, glob, subprocess, numpy as np, torch
sys.path.insert(0, "E:/dualcam_project/src")
from dualcam_bev_lss import DualCamBEVLSS
from render_prof_style import render_bev_prof, fov_visible_mask

ROOT="E:/dualcam_project"; DATA=f"{ROOT}/data"; CK=f"{ROOT}/checkpoints"
D=f"{DATA}/carla_dualcam_t01_temporal"
OUT=f"{ROOT}/results"; os.makedirs(OUT, exist_ok=True)
FFMPEG="E:/miniforge3/envs/lss/lib/site-packages/imageio_ffmpeg/binaries/ffmpeg-win-x86_64-v7.1.exe"
MODEL=f"{CK}/dualcam_bev_lss_9set_depth_bv3_best.pt"
dev="cuda" if torch.cuda.is_available() else "cpu"
IMG_H,IMG_W=224,400
BEV=240; RES,X0,Y0=0.25,-30.0,-30.0
STATIC=(3,4,5)  # road, sidewalk, lane -> safe to remember; vehicle/ped are not
VIS=fov_visible_mask(BEV,BEV)          # 90-deg visible mask
BLIND=~VIS

# ---- ego-motion warp (same convention as ego_xy_to_world_loc) ----
def px_to_ego(col,row): return (X0+(BEV-1-row)*RES, Y0+col*RES)     # -> (x fwd, y right)
def ego_to_px(x,y):     return ((y-Y0)/RES, (BEV-1)-(x-X0)/RES)     # -> (col,row)
def world_from_ego(x,y,ex,ey,yaw_deg):
    a=math.radians(yaw_deg); c,s=math.cos(a),math.sin(a)
    return ex+c*x-s*y, ey+s*x+c*y
def ego_from_world(wx,wy,ex,ey,yaw_deg):
    a=math.radians(yaw_deg); c,s=math.cos(a),math.sin(a); dx,dy=wx-ex,wy-ey
    return c*dx+s*dy, -s*dx+c*dy
def align_M(pose_cur, pose_prev):
    """affine mapping CURRENT-frame px -> PREV-frame px (for WARP_INVERSE_MAP)."""
    src=[(10,10),(BEV-10,10),(10,BEV-10)]; dst=[]
    for col,row in src:
        x,y=px_to_ego(col,row); wx,wy=world_from_ego(x,y,*pose_cur)
        x2,y2=ego_from_world(wx,wy,*pose_prev); c2,r2=ego_to_px(x2,y2); dst.append((c2,r2))
    return cv2.getAffineTransform(np.float32(src), np.float32(dst))

# ---- model ----
meta=torch.load(MODEL,map_location="cpu",weights_only=False)
ds=meta.get("lss_downsample",8); pool=meta.get("lss_pool",0.5)
m=DualCamBEVLSS.from_calib_file(f"{D}/calibration.json",num_classes=6,bev_h=240,bev_w=240,
                                img_h=IMG_H,img_w=IMG_W,downsample=ds,pool_res=pool,pretrained=False)
st=meta["model_state_dict"] if isinstance(meta,dict) and "model_state_dict" in meta else meta
m.load_state_dict(st,strict=True); m=m.to(dev).eval()
def readimg(p):
    im=cv2.imread(p); im=cv2.resize(im,(IMG_W,IMG_H))[:,:,::-1].copy().astype(np.float32)/255.0
    return torch.from_numpy(im).permute(2,0,1)
def predict(n):
    x=torch.stack([readimg(f"{D}/front_rgb/{n}"),readimg(f"{D}/rear_rgb/{n}")],0).unsqueeze(0).to(dev)
    with torch.no_grad(): return torch.argmax(m(x)[0],0).cpu().numpy().astype(np.uint8)

poses=json.load(open(f"{D}/poses.json"))
files=sorted(os.path.basename(p) for p in glob.glob(f"{D}/bev_gt/*.png"))
files=[f for f in files if f in poses]
print(f"{len(files)} frames with poses")

def titled(bev,t):
    bar=np.zeros((32,bev.shape[1],3),np.uint8); cv2.putText(bar,t,(8,22),cv2.FONT_HERSHEY_SIMPLEX,0.62,(255,255,255),2)
    return np.vstack([bar,bev])
def rgb_panel(p,t,w=520):
    im=cv2.resize(cv2.imread(p),(w,int(w*9/16))); cv2.putText(im,t,(10,26),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2); return im

MAXAGE=20   # rolling window: forget blind-zone memory older than this many frames
memory=np.zeros((BEV,BEV),np.uint8)
age=np.full((BEV,BEV),9999,np.int32)   # frames since a cell was last seen in-FOV
prev_pose=None
frames_out=[]
P=490
def clean_static(lab):
    """close gaps + drop tiny islands on the static map so warped memory stays tidy."""
    out=np.zeros_like(lab)
    for cls in STATIC:
        mm=cv2.morphologyEx((lab==cls).astype(np.uint8)*255, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))
        out[mm>0]=cls
    return out
for i,n in enumerate(files):
    pred=predict(n); pose=poses[n]
    if prev_pose is not None:
        M=align_M(pose, prev_pose)
        memory=cv2.warpAffine(memory,M,(BEV,BEV),flags=cv2.INTER_NEAREST|cv2.WARP_INVERSE_MAP,borderValue=0)
        age=cv2.warpAffine(age.astype(np.float32),M,(BEV,BEV),flags=cv2.INTER_NEAREST|cv2.WARP_INVERSE_MAP,borderValue=9999).astype(np.int32)
    age=age+1
    # fresh observation in the visible region resets memory + age there
    memory[VIS]=np.where(np.isin(pred[VIS],STATIC),pred[VIS],0).astype(np.uint8)
    age[VIS]=0
    memory[age>MAXAGE]=0                     # forget stale blind-zone memory
    memory=clean_static(memory)              # keep the remembered static map tidy
    # fuse for display: remembered static fills blind, current pred owns the whole visible region
    fused=memory.copy()
    fused[VIS]=pred[VIS]
    prev_pose=pose

    gt=cv2.imread(f"{D}/bev_gt/{n}",cv2.IMREAD_GRAYSCALE)
    cur_panel = titled(render_bev_prof(pred, out_size=P, dim_blind=BLIND, smooth_road=True, min_vehicle=15, min_lane=6), "Current frame only")
    fus_panel = titled(render_bev_prof(fused,out_size=P, dim_blind=BLIND, smooth_road=True, min_vehicle=15, min_lane=6), "Temporal fusion")
    gt_panel  = titled(render_bev_prof(gt,   out_size=P, dim_blind=BLIND), "Ground Truth")
    bevrow=np.hstack([cur_panel,fus_panel,gt_panel]); W=bevrow.shape[1]
    inrow=np.hstack([rgb_panel(f"{D}/front_rgb/{n}","Front RGB"),rgb_panel(f"{D}/rear_rgb/{n}","Rear RGB")])
    pad=(W-inrow.shape[1])//2; inrow=cv2.copyMakeBorder(inrow,0,0,pad,W-inrow.shape[1]-pad,cv2.BORDER_CONSTANT,value=0)
    cap=np.zeros((28,W,3),np.uint8); cv2.putText(cap,f"Town01 temporal BEV fusion  frame {n}  -  blind zone filled from ego-motion-warped memory (static only)",(10,19),cv2.FONT_HERSHEY_SIMPLEX,0.5,(180,180,180),1)
    frames_out.append(np.vstack([inrow,np.zeros((8,W,3),np.uint8),bevrow,cap]))
    if i%50==0: print(f"{i}/{len(files)}",flush=True)

h,w=frames_out[0].shape[:2]
tmp=f"{OUT}/_temporal_tmp.mp4"; vw=cv2.VideoWriter(tmp,cv2.VideoWriter_fourcc(*"mp4v"),12,(w,h))
for f in frames_out: vw.write(f)
vw.release()
final=f"{OUT}/TEMPORAL_fusion_town01.mp4"
subprocess.run([FFMPEG,"-y","-i",tmp,"-c:v","libx264","-pix_fmt","yuv420p","-crf","20",final],check=True,capture_output=True)
os.remove(tmp)
# a few PNGs
for fr in [90,180,240,300]:
    if fr<len(frames_out): cv2.imwrite(f"{OUT}/_temporal_{fr}.png", frames_out[fr])
print("DONE:",final,f"({w}x{h})")
